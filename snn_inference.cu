// 包含必要的头文件
#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <algorithm>
#include <cmath>
#include <cuda_runtime.h>
#include <cfloat>

// 定义全局配置参数
#define TIME_WINDOW 80.0f      // 定义总时间窗口
#define VTH_INIT 1.0f          // 定义初始阈值
#define INF_TIME 9999.0f       // 定义未发放的时间标志
#define MAX_CONN_LIMIT 4096    // 定义显存缓冲区的最大连接限制

// 检查CUDA调用是否成功的宏定义
#define CHECK_CUDA(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            printf("CUDA Error: %s at line %d\n", cudaGetErrorString(err), __LINE__); \
            exit(1); \
        } \
    } while (0)

// 定义权重结构体
struct LayerWeights {
    float* d_kernel; // 指向设备端的卷积核权重
    float* d_bias;   // 指向设备端的偏置
    float tc_fire;   // 发放时间常数
    float td;        // 延迟时间
    int k_h, k_w;    // 卷积核的高度和宽度
    int c_in, c_out; // 输入通道数和输出通道数
};

// 定义脉冲事件结构体
struct InputEvent {
    float time;   // 脉冲到达时间
    float weight; // 脉冲权重
};

// CUDA核函数：图像编码，将像素值转换为脉冲时间
__global__ void k_encode_image(const float* img, float* out_spikes, int size, float tc, float td) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x; // 计算线程索引
    if (idx >= size) return; // 检查索引是否越界

    float pixel = img[idx]; // 获取当前像素值
    if (pixel < 1e-5) { // 如果像素值过暗或为零
        out_spikes[idx] = INF_TIME; // 设置为未发放时间
        return;
    }

    float t_float = td - tc * logf(pixel); // 使用TTFS编码计算时间
    if (t_float < 0.0f) t_float = 0.0f; // 钳位时间为非负值
    float t_spike = ceilf(t_float); // 向上取整为离散时间步

    if (t_spike > TIME_WINDOW) out_spikes[idx] = INF_TIME; // 超出时间窗口
    else out_spikes[idx] = t_spike; // 设置脉冲时间
}

// CUDA核函数：核心推理层，模拟神经元的时间积分过程
__global__ void k_layer_inference(
    const float* input_spikes,    // 上一层的输出脉冲时间
    float* output_spikes,         // 当前层的输出脉冲时间
    const float* kernel,          // 当前层的权重
    const float* bias,            // 当前层的偏置
    InputEvent* global_event_buffer, // 全局事件缓冲区
    int H_in, int W_in, int C_in, // 输入特征图的维度
    int H_out, int W_out, int C_out, // 输出特征图的维度
    int K_h, int K_w,             // 卷积核的尺寸
    float tc_fire,  // 当前层的发放时间常数
    float tc_integ, // 上一层的积分时间常数
    float td_fire,  // 当前层的发放延迟时间
    float td_integ, // 上一层的积分延迟时间
    bool is_fc                    // 是否为全连接层
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x; // 计算线程索引
    int total_neurons = H_out * W_out * C_out; // 计算总神经元数
    if (idx >= total_neurons) return; // 检查索引是否越界

    InputEvent* my_events = global_event_buffer + idx * MAX_CONN_LIMIT; // 获取当前神经元的事件缓冲区
    int event_count = 0; // 初始化事件计数器

    int c_out, w_out, h_out; // 定义输出通道和坐标
    if (is_fc) {
        c_out = idx; w_out = 0; h_out = 0; // 全连接层的特殊处理
    } else {
        c_out = idx % C_out;
        w_out = (idx / C_out) % W_out;
        h_out = (idx / C_out) / W_out;
    }

    float b_val = bias[c_out]; // 获取偏置值

    // 收集所有输入事件
    if (is_fc) {
        for (int i = 0; i < C_in; ++i) {
            float t_in = input_spikes[i];
            if (t_in < INF_TIME) {
                float w = kernel[i * C_out + c_out];
                float t_rel_in = t_in - td_integ; // 计算相对时间
                float w_decayed = w * expf(-t_rel_in / tc_integ); // 权重衰减

                if (event_count < MAX_CONN_LIMIT) {
                    my_events[event_count] = {t_in, w_decayed}; // 存储事件
                    event_count++;
                }
            }
        }
    } else {
        // 卷积层逻辑
        int h_in_start = h_out; // 输入起始高度
        int w_in_start = w_out; // 输入起始宽度
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

                        float t_rel_in = t_in - td_integ; // 计算相对时间
                        float w_decayed = w * expf(-t_rel_in / tc_integ); // 权重衰减

                        if (event_count < MAX_CONN_LIMIT) {
                            my_events[event_count] = {t_in, w_decayed}; // 存储事件
                            event_count++;
                        }
                    }
                }
            }
        }
    }

    // 按时间排序事件
    for (int i = 0; i < event_count - 1; ++i) {
        for (int j = 0; j < event_count - i - 1; ++j) {
            if (my_events[j].time > my_events[j+1].time) {
                InputEvent temp = my_events[j];
                my_events[j] = my_events[j+1];
                my_events[j+1] = temp;
            }
        }
    }

    // 模拟时间流逝并进行积分
    float v_mem_acc = 0.0f; // 初始化膜电位累积值
    float final_spike_time = INF_TIME; // 初始化最终发放时间
    float T_start = td_fire; // 发放起始时间
    int current_event_idx = 0; // 当前事件索引
    int t_start_int = 0;
    if (T_start > 0) t_start_int = (int)ceilf(T_start);

    for (int t = t_start_int; t < (int)TIME_WINDOW; ++t) {
        float t_float = (float)t;

        while (current_event_idx < event_count && my_events[current_event_idx].time <= t_float) {
            v_mem_acc += my_events[current_event_idx].weight; // 累加权重
            current_event_idx++;
        }

        float v_mem = v_mem_acc + b_val; // 计算当前膜电位
        float time_rel = t_float - T_start; // 计算相对时间
        float v_th = VTH_INIT * expf(-time_rel / tc_fire); // 计算阈值

        if (v_mem >= v_th && v_th >= 1e-5) {
            final_spike_time = t_float; // 记录发放时间
            break;
        }
    }

    output_spikes[idx] = final_spike_time; // 设置输出脉冲时间
}

// CUDA核函数：池化层，进行最小时间池化
__global__ void k_pooling(const float* in_spikes, float* out_spikes, int H_in, int W_in, int C) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x; // 计算线程索引
    int H_out = H_in / 2; // 输出高度
    int W_out = W_in / 2; // 输出宽度
    int total = H_out * W_out * C; // 计算总输出大小
    if (idx >= total) return; // 检查索引是否越界

    int c = idx % C; // 当前通道
    int w = (idx / C) % W_out; // 当前宽度
    int h = (idx / C) / W_out; // 当前高度
    int h_start = h * 2; // 输入起始高度
    int w_start = w * 2; // 输入起始宽度
    float min_t = INF_TIME; // 初始化最小时间

    // 2x2最大池化，在TTFS中等价于最小时间
    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 2; ++j) {
            int cur_h = h_start + i;
            int cur_w = w_start + j;
            if (cur_h < H_in && cur_w < W_in) {
                float t = in_spikes[(cur_h * W_in + cur_w) * C + c];
                if (t < min_t) min_t = t; // 更新最小时间
            }
        }
    }
    out_spikes[idx] = min_t; // 设置输出脉冲时间
}

// 主机端辅助函数：加载权重
void load_weights(const std::string& filename, std::vector<LayerWeights>& layers) {
    std::ifstream f(filename, std::ios::binary);
    if (!f.is_open()) { std::cerr << "File not found: " << filename << "\n"; exit(1); }
    int num_layers;
    f.read((char*)&num_layers, 4); // 读取层数
    for(int i=0; i<num_layers; ++i) {
        LayerWeights l;
        int ndim;
        // Kernel
        f.read((char*)&ndim, 4); // 读取维度
        std::vector<int> k_shape(ndim);
        int k_size = 1;
        for(int j=0; j<ndim; ++j) {
            f.read((char*)&k_shape[j], 4); // 读取每一维的大小
            k_size *= k_shape[j]; // 计算总大小
        }
        if (ndim == 4) { l.k_h = k_shape[0]; l.k_w = k_shape[1]; l.c_in = k_shape[2]; l.c_out = k_shape[3]; }
        else { l.k_h = 1; l.k_w = 1; l.c_in = k_shape[0]; l.c_out = k_shape[1]; }
        
        std::vector<float> h_kernel(k_size);
        f.read((char*)h_kernel.data(), k_size * 4); // 读取权重数据
        CHECK_CUDA(cudaMalloc(&l.d_kernel, k_size * 4));
        CHECK_CUDA(cudaMemcpy(l.d_kernel, h_kernel.data(), k_size * 4, cudaMemcpyHostToDevice));

        // Bias
        f.read((char*)&ndim, 4);
        int b_size;
        f.read((char*)&b_size, 4); // 读取偏置大小
        std::vector<float> h_bias(b_size);
        f.read((char*)h_bias.data(), b_size * 4); // 读取偏置数据
        CHECK_CUDA(cudaMalloc(&l.d_bias, b_size * 4));
        CHECK_CUDA(cudaMemcpy(l.d_bias, h_bias.data(), b_size * 4, cudaMemcpyHostToDevice));

        // TC / TD
        f.read((char*)&ndim, 4); int tc_size; f.read((char*)&tc_size, 4);
        f.read((char*)&l.tc_fire, 4); // 读取发放时间常数

        f.read((char*)&ndim, 4); int td_size; f.read((char*)&td_size, 4);
        f.read((char*)&l.td, 4); // 读取延迟时间
        layers.push_back(l); // 将层信息添加到向量中
    }
}

int main(int argc, char* argv[]) {
    // 解析命令行参数，获取目标图片索引，默认为4
    int target_idx = 4;
    if (argc > 1) target_idx = std::atoi(argv[1]);
    std::cout << ">>> Target Image Index: " << target_idx << std::endl;

    // 加载神经网络的权重
    std::vector<LayerWeights> layers;
    load_weights("exported_models/snn_weights.bin", layers);

    // 加载目标图片数据
    std::string img_path = "dataset_downloaded/mnist_float/" + std::to_string(target_idx) + ".bin";
    std::ifstream img_f(img_path, std::ios::binary);
    if (!img_f.is_open()) {
        std::cerr << "Image file not found: " << img_path << "\n";
        return 1;
    }
    std::vector<float> h_img(784);
    img_f.read((char*)h_img.data(), 784 * 4);

    // 分配显存并将图片数据拷贝到设备端
    float *d_img, *d_s0, *d_s1, *d_s1_p, *d_s2, *d_s2_p, *d_out;
    CHECK_CUDA(cudaMalloc(&d_img, 784 * 4));
    CHECK_CUDA(cudaMemcpy(d_img, h_img.data(), 784 * 4, cudaMemcpyHostToDevice));

    // 分配特征图的显存空间
    CHECK_CUDA(cudaMalloc(&d_s0, 28*28*1*4)); 
    CHECK_CUDA(cudaMalloc(&d_s1, 24*24*12*4));
    CHECK_CUDA(cudaMalloc(&d_s1_p, 12*12*12*4));
    CHECK_CUDA(cudaMalloc(&d_s2, 8*8*64*4));
    CHECK_CUDA(cudaMalloc(&d_s2_p, 4*4*64*4));
    CHECK_CUDA(cudaMalloc(&d_out, 10*4));

    // 分配全局事件缓冲区的显存空间
    InputEvent* d_event_buffer;
    CHECK_CUDA(cudaMalloc(&d_event_buffer, 6912 * MAX_CONN_LIMIT * sizeof(InputEvent)));

    std::cout << ">>> Starting SNN Inference (Parallel Logic)..." << std::endl;
    float tc_in = 17.452274f; // 输入层的时间常数
    float td_in = 0.0f;       // 输入层的延迟时间

    // 图像编码：将像素值转换为脉冲时间
    k_encode_image<<<1, 784>>>(d_img, d_s0, 784, 17.452274f, 0.0f);
    CHECK_CUDA(cudaDeviceSynchronize());

    // 卷积层1的推理
    int threads_c1 = 24 * 24 * 12; 
    k_layer_inference<<<(threads_c1+255)/256, 256>>>(
        d_s0, d_s1, layers[0].d_kernel, layers[0].d_bias, d_event_buffer,
        28, 28, 1, 24, 24, 12, 5, 5, 
        layers[0].tc_fire, // 当前层的时间常数
        tc_in,             // 上一层的时间常数
        layers[0].td,      // 当前层的延迟时间
        td_in,             // 上一层的延迟时间
        false
    );
    CHECK_CUDA(cudaDeviceSynchronize());

    // 池化层1的推理
    int threads_p1 = 12 * 12 * 12;
    k_pooling<<<(threads_p1+255)/256, 256>>>(d_s1, d_s1_p, 24, 24, 12);

    // 卷积层2的推理
    int threads_c2 = 8 * 8 * 64;
    k_layer_inference<<<(threads_c2+255)/256, 256>>>(
        d_s1_p, d_s2, layers[1].d_kernel, layers[1].d_bias, d_event_buffer,
        12, 12, 12, 8, 8, 64, 5, 5,
        layers[1].tc_fire, // 当前层的时间常数
        layers[0].tc_fire, // 上一层的时间常数
        layers[1].td,      // 当前层的延迟时间
        layers[0].td,      // 上一层的延迟时间
        false
    );
    CHECK_CUDA(cudaDeviceSynchronize());

    // 池化层2的推理
    int threads_p2 = 4 * 4 * 64; 
    k_pooling<<<(threads_p2+255)/256, 256>>>(d_s2, d_s2_p, 8, 8, 64);

    // 全连接层的推理
    k_layer_inference<<<1, 10>>>(
        d_s2_p, d_out, layers[2].d_kernel, layers[2].d_bias, d_event_buffer,
        1024, 1, 1024, 1, 1, 10, 1, 1,
        layers[2].tc_fire, // 当前层的时间常数
        layers[1].tc_fire, // 上一层的时间常数
        layers[2].td,      // 当前层的延迟时间
        layers[1].td,      // 上一层的延迟时间
        true
    );
    CHECK_CUDA(cudaDeviceSynchronize());

    // 获取推理结果并拷贝回主机端
    std::vector<float> h_out(10);
    CHECK_CUDA(cudaMemcpy(h_out.data(), d_out, 10 * 4, cudaMemcpyDeviceToHost));

    std::cout << "\n>>> Inference Result (Spike Times):\n";
    float min_time = INF_TIME; // 初始化最小时间
    int pred_cls = -1;         // 初始化预测类别

    // 输出每个类别的脉冲时间并确定预测类别
    for (int i = 0; i < 10; ++i) {
        if (h_out[i] < INF_TIME)
            printf("  Class %d: %.2f\n", i, h_out[i]);
        else
            printf("  Class %d: Did not fire\n", i);

        if (h_out[i] < min_time) { 
            min_time = h_out[i]; 
            pred_cls = i; 
        }
    }

    std::cout << "--------------------------------\n";
    std::cout << "Prediction: " << pred_cls << "\n";

    // 验证预测结果是否正确
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