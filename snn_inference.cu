#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <algorithm>
#include <cmath>
#include <cuda_runtime.h>
#include <cfloat>

// ==========================================
// 全局配置
// ==========================================
#define TIME_WINDOW 80.0f      // 总时间窗口
#define VTH_INIT 1.0f           // 初始阈值 V0
#define INF_TIME 9999.0f        // 表示未发放
// 显存缓冲区大小：必须足够容纳最大的 Fan-in (FC层=1024)
// 设为 4096 保证安全
#define MAX_CONN_LIMIT 4096     

#define CHECK_CUDA(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            printf("CUDA Error: %s at line %d\n", cudaGetErrorString(err), __LINE__); \
            exit(1); \
        } \
    } while (0)

// 权重结构体
struct LayerWeights {
    float* d_kernel;
    float* d_bias;
    float tc_fire; // 修改名称：明确这是“发放”时间常数
    float td;
    int k_h, k_w, c_in, c_out;
};

// 脉冲事件结构体
struct InputEvent {
    float time;   // 到达时间
    float weight; // 权重
};

// ==========================================
// CUDA Kernels
// ==========================================

// Kernel 1: 图像编码 (Pixel -> Spike Time)
__global__ void k_encode_image(const float* img, float* out_spikes, int size, float tc, float td) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    
    float pixel = img[idx];
    // 像素过暗或者为0，视为无穷晚
    if (pixel < 1e-5) { 
        out_spikes[idx] = INF_TIME; 
        return; 
    }
    
    // TTFS 编码: t = td - tau * ln(pixel)
    float t_float = td - tc * logf(pixel);
    
    // 钳位到 0 (不能是负时间)
    if (t_float < 0.0f) t_float = 0.0f;
    
    // 向上取整得到离散时间步
    float t_spike = ceilf(t_float);
    
    if (t_spike > TIME_WINDOW) out_spikes[idx] = INF_TIME;
    else out_spikes[idx] = t_spike;
}

// Kernel 2: 核心推理层 (并行时间积分)
// 逻辑：收集所有输入 -> 排序 -> 模拟时间流逝进行积分
__global__ void k_layer_inference(
    const float* input_spikes,    // 上一层输出
    float* output_spikes,         // 本层输出
    const float* kernel,          // 权重
    const float* bias,            // 偏置
    InputEvent* global_event_buffer, // 全局显存缓冲 (防栈溢出)
    int H_in, int W_in, int C_in,
    int H_out, int W_out, int C_out,
    int K_h, int K_w,             // Kernel Size
    float tc_fire,  // [当前层] 发放 TC (用于计算 Vth)
    float tc_integ, // [上一层] 发放 TC (用于积分衰减)
    float td_fire,  // [当前层] 发放 TD (用于计算 Vth)
    float td_integ, // [上一层] 发放 TD (用于积分补偿) <--- 新增参数
    bool is_fc                    // 是否为全连接层
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_neurons = H_out * W_out * C_out;
    if (idx >= total_neurons) return;

    // 1. 定位当前神经元的事件缓冲区
    InputEvent* my_events = global_event_buffer + idx * MAX_CONN_LIMIT;
    int event_count = 0;

    // 解析坐标
    int c_out, w_out, h_out;
    if (is_fc) {
        c_out = idx; w_out = 0; h_out = 0;
    } else {
        c_out = idx % C_out;
        w_out = (idx / C_out) % W_out;
        h_out = (idx / C_out) / W_out;
    }

    float b_val = bias[c_out];

    // ==========================================
    // Phase A: 收集 (Gather) - 找出所有会发放的输入
    // ==========================================
    float bias_val = bias[c_out];

    if (is_fc) {
        for (int i = 0; i < C_in; ++i) {
            float t_in = input_spikes[i];
            if (t_in < INF_TIME) {
                float w = kernel[i * C_out + c_out];
                
                // [FIX]: 减去上一层的 td (td_integ)
                // 还原真实的相对时间，确保 exp(ln(activation)) 正确还原
                float t_rel_in = t_in - td_integ; 
                float w_decayed = w * expf(-t_rel_in / tc_integ);

                if (event_count < MAX_CONN_LIMIT) {
                    my_events[event_count] = {t_in, w_decayed}; 
                    event_count++;
                }
            }
        }
    } else {
        // Conv 层逻辑
        int h_in_start = h_out; // Stride=1
        int w_in_start = w_out;
        for (int kh = 0; kh < K_h; ++kh) {
            for (int kw = 0; kw < K_w; ++kw) {
                for (int cin = 0; cin < C_in; ++cin) {
                    int h_in = h_in_start + kh;
                    int w_in = w_in_start + kw;
                    int in_idx = (h_in * W_in + w_in) * C_in + cin;
                    
                    float t_in = input_spikes[in_idx];
                    if (t_in < INF_TIME) {
                        int w_idx = ((kh * K_w + kw) * C_in + cin) * C_out + c_out;
                        float w = kernel[w_idx];
                        
                        // [FIX]: 减去上一层的 td
                        float t_rel_in = t_in - td_integ;
                        float w_decayed = w * expf(-t_rel_in / tc_integ);

                        if (event_count < MAX_CONN_LIMIT) {
                            my_events[event_count] = {t_in, w_decayed};
                            event_count++;
                        }
                    }
                }
            }
        }
    }

    // ==========================================
    // Phase B: 排序 (Sort) - 按时间先后排列
    // ==========================================
    for (int i = 0; i < event_count - 1; ++i) {
        for (int j = 0; j < event_count - i - 1; ++j) {
            if (my_events[j].time > my_events[j+1].time) {
                InputEvent temp = my_events[j];
                my_events[j] = my_events[j+1];
                my_events[j+1] = temp;
            }
        }
    }

    // ==========================================
    // Phase C: 积分 (Integrate) - 模拟时间流逝
    // ==========================================
    float v_mem_acc = 0.0f;
    float final_spike_time = INF_TIME;
    
    // 发放起始时间由当前层的 td 决定
    float T_start = td_fire; 
    int current_event_idx = 0;
    int t_start_int = 0;
    if (T_start > 0) t_start_int = (int)ceilf(T_start);

    for (int t = t_start_int; t < (int)TIME_WINDOW; ++t) {
        float t_float = (float)t;

        while (current_event_idx < event_count && my_events[current_event_idx].time <= t_float) {
            v_mem_acc += my_events[current_event_idx].weight;
            current_event_idx++;
        }

        // Bias 直接加 (不乘时间)
        float v_mem = v_mem_acc + b_val;

        // 计算当前层阈值 (使用 tc_fire 和 td_fire)
        float time_rel = t_float - T_start;
        float v_th = VTH_INIT * expf(-time_rel / tc_fire);

        if (v_mem >= v_th && v_th >= 1e-5) {
            final_spike_time = t_float;
            break; 
        }
    }
    
    output_spikes[idx] = final_spike_time;
}

// Kernel 3: 池化层 (Min Time Pooling)
__global__ void k_pooling(const float* in_spikes, float* out_spikes, int H_in, int W_in, int C) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int H_out = H_in / 2;
    int W_out = W_in / 2;
    int total = H_out * W_out * C;
    if (idx >= total) return;

    int c = idx % C;
    int w = (idx / C) % W_out;
    int h = (idx / C) / W_out;
    int h_start = h * 2;
    int w_start = w * 2;
    float min_t = INF_TIME;

    // 2x2 Max Pooling -> 在 TTFS 中等于 Min Time
    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 2; ++j) {
            int cur_h = h_start + i;
            int cur_w = w_start + j;
            if (cur_h < H_in && cur_w < W_in) {
                float t = in_spikes[(cur_h * W_in + cur_w) * C + c];
                if (t < min_t) min_t = t;
            }
        }
    }
    out_spikes[idx] = min_t;
}

// ==========================================
// Host 端辅助函数
// ==========================================
void load_weights(const std::string& filename, std::vector<LayerWeights>& layers) {
    std::ifstream f(filename, std::ios::binary);
    if (!f.is_open()) { std::cerr << "File not found: " << filename << "\n"; exit(1); }
    int num_layers;
    f.read((char*)&num_layers, 4);
    for(int i=0; i<num_layers; ++i) {
        LayerWeights l;
        int ndim;
        // Kernel
        f.read((char*)&ndim, 4);
        std::vector<int> k_shape(ndim);
        int k_size = 1;
        for(int j=0; j<ndim; ++j) {
            f.read((char*)&k_shape[j], 4);
            k_size *= k_shape[j];
        }
        if (ndim == 4) { l.k_h = k_shape[0]; l.k_w = k_shape[1]; l.c_in = k_shape[2]; l.c_out = k_shape[3]; }
        else { l.k_h = 1; l.k_w = 1; l.c_in = k_shape[0]; l.c_out = k_shape[1]; }
        
        std::vector<float> h_kernel(k_size);
        f.read((char*)h_kernel.data(), k_size * 4);
        CHECK_CUDA(cudaMalloc(&l.d_kernel, k_size * 4));
        CHECK_CUDA(cudaMemcpy(l.d_kernel, h_kernel.data(), k_size * 4, cudaMemcpyHostToDevice));

        // Bias
        f.read((char*)&ndim, 4);
        int b_size;
        f.read((char*)&b_size, 4);
        std::vector<float> h_bias(b_size);
        f.read((char*)h_bias.data(), b_size * 4);
        CHECK_CUDA(cudaMalloc(&l.d_bias, b_size * 4));
        CHECK_CUDA(cudaMemcpy(l.d_bias, h_bias.data(), b_size * 4, cudaMemcpyHostToDevice));

        // TC / TD
        f.read((char*)&ndim, 4); int tc_size; f.read((char*)&tc_size, 4);
        f.read((char*)&l.tc_fire, 4); // <--- 修改为 l.tc_fire

        f.read((char*)&ndim, 4); int td_size; f.read((char*)&td_size, 4);
        f.read((char*)&l.td, 4);
        layers.push_back(l);
    }
}

int main(int argc, char* argv[]) {
    // 0. 参数解析
    int target_idx = 4; // 默认图片
    if (argc > 1) target_idx = std::atoi(argv[1]);
    std::cout << ">>> Target Image Index: " << target_idx << std::endl;

    // 1. 加载权重
    std::vector<LayerWeights> layers;
    load_weights("exported_models/snn_weights.bin", layers);
    
    // 2. 加载图片
    std::string img_path = "dataset_downloaded/mnist_float/" + std::to_string(target_idx) + ".bin";
    std::ifstream img_f(img_path, std::ios::binary);
    if (!img_f.is_open()) { std::cerr << "Image file not found: " << img_path << "\n"; return 1; }
    std::vector<float> h_img(784);
    img_f.read((char*)h_img.data(), 784 * 4);
    
    // 3. 分配显存
    float *d_img, *d_s0, *d_s1, *d_s1_p, *d_s2, *d_s2_p, *d_out;
    CHECK_CUDA(cudaMalloc(&d_img, 784 * 4));
    CHECK_CUDA(cudaMemcpy(d_img, h_img.data(), 784 * 4, cudaMemcpyHostToDevice));

    // Feature Maps
    CHECK_CUDA(cudaMalloc(&d_s0, 28*28*1*4)); 
    CHECK_CUDA(cudaMalloc(&d_s1, 24*24*12*4));
    CHECK_CUDA(cudaMalloc(&d_s1_p, 12*12*12*4));
    CHECK_CUDA(cudaMalloc(&d_s2, 8*8*64*4));
    CHECK_CUDA(cudaMalloc(&d_s2_p, 4*4*64*4));
    CHECK_CUDA(cudaMalloc(&d_out, 10*4));

    // 全局事件缓冲区
    InputEvent* d_event_buffer;
    CHECK_CUDA(cudaMalloc(&d_event_buffer, 6912 * MAX_CONN_LIMIT * sizeof(InputEvent)));

    std::cout << ">>> Starting SNN Inference (Parallel Logic)..." << std::endl;
    float tc_in = 17.452274f;
    float td_in = 0.0f; // Input Layer 的 TD 通常为 0

    // Step A: Encode
    k_encode_image<<<1, 784>>>(d_img, d_s0, 784, 17.452274f, 0.0f);
    CHECK_CUDA(cudaDeviceSynchronize());

    // Step B: Conv1
    int threads_c1 = 24 * 24 * 12; 
    k_layer_inference<<<(threads_c1+255)/256, 256>>>(
        d_s0, d_s1, layers[0].d_kernel, layers[0].d_bias, d_event_buffer,
        28, 28, 1, 24, 24, 12, 5, 5, 
        layers[0].tc_fire, // tc_fire (Conv1)
        tc_in,             // tc_integ (Input)
        layers[0].td,      // td_fire (Conv1)
        td_in,             // td_integ (Input) <--- 新增
        false
    );
    CHECK_CUDA(cudaDeviceSynchronize());

    // Step C: Pool1
    int threads_p1 = 12 * 12 * 12;
    k_pooling<<<(threads_p1+255)/256, 256>>>(d_s1, d_s1_p, 24, 24, 12);

    // Step D: Conv2
    int threads_c2 = 8 * 8 * 64;
    k_layer_inference<<<(threads_c2+255)/256, 256>>>(
        d_s1_p, d_s2, layers[1].d_kernel, layers[1].d_bias, d_event_buffer,
        12, 12, 12, 8, 8, 64, 5, 5,
        layers[1].tc_fire, // tc_fire (Conv2)
        layers[0].tc_fire, // tc_integ (Conv1)
        layers[1].td,      // td_fire (Conv2)
        layers[0].td,      // td_integ (Conv1) <--- 新增
        false
    );
    CHECK_CUDA(cudaDeviceSynchronize());

    // Step E: Pool2
    int threads_p2 = 4 * 4 * 64; 
    k_pooling<<<(threads_p2+255)/256, 256>>>(d_s2, d_s2_p, 8, 8, 64);

    // Step F: FC1
    k_layer_inference<<<1, 10>>>(
        d_s2_p, d_out, layers[2].d_kernel, layers[2].d_bias, d_event_buffer,
        1024, 1, 1024, 1, 1, 10, 1, 1,
        layers[2].tc_fire, // tc_fire (FC1)
        layers[1].tc_fire, // tc_integ (Conv2)
        layers[2].td,      // td_fire (FC1)
        layers[1].td,      // td_integ (Conv2) <--- 新增
        true
    );
    CHECK_CUDA(cudaDeviceSynchronize());
    
    // 4. 获取结果
    std::vector<float> h_out(10);
    CHECK_CUDA(cudaMemcpy(h_out.data(), d_out, 10 * 4, cudaMemcpyDeviceToHost));

    std::cout << "\n>>> Inference Result (Spike Times):\n";
    float min_time = INF_TIME; // 修正变量名
    int pred_cls = -1;

    for (int i = 0; i < 10; ++i) {
        if (h_out[i] < INF_TIME)
            printf("  Class %d: %.2f\n", i, h_out[i]);
        else
            printf("  Class %d: Did not fire\n", i);
            
        // 修正这里：使用 min_time 而不是 min_t
        if (h_out[i] < min_time) { 
            min_time = h_out[i]; 
            pred_cls = i; 
        }
    }
    
    std::cout << "--------------------------------\n";
    std::cout << "Prediction: " << pred_cls << "\n";
    
    // 读取 Label 进行验证
    std::ifstream lbl_f("dataset_downloaded/mnist_float/label_onehot", std::ios::binary);
    std::vector<float> lbl(10);
    lbl_f.seekg(target_idx * 10 * 4, std::ios::beg);
    lbl_f.read((char*)lbl.data(), 10*4);
    
    int true_cls = -1;
    for(int i=0;i<10;++i) if(lbl[i]>0.5) true_cls = i;
    std::cout << "Ground Truth: " << true_cls << std::endl;

    if (pred_cls == true_cls) std::cout << "SUCCESS!\n";
    else std::cout << "FAILED.\n";

    return 0;
}