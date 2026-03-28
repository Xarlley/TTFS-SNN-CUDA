#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <algorithm>
#include <cmath>
#include <cuda_runtime.h>
#include <cfloat>

#define TIME_WINDOW 80.0f
#define TIME_FIRE_START 40.0f  
#define VTH_INIT 1.0f
#define INF_TIME 9999.0f
// ✅ 新增：用于标记被“预测剔除机制”在第 0 步就干掉的神经元
#define PRUNED_TIME 9998.0f 

#define CHECK_CUDA(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            printf("CUDA Error: %s at line %d\n", cudaGetErrorString(err), __LINE__); \
            exit(1); \
        } \
    } while (0)

struct LayerWeights {
    float* d_kernel; float* d_bias;
    float tc_fire; float td;
    int k_h, k_w, c_in, c_out;
};

__global__ void k_encode_image(const float* img, float* out_spikes, int size, float tc, float td) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    float pixel = img[idx];
    if (pixel < 1e-5) { out_spikes[idx] = INF_TIME; return; }
    float t_float = td - tc * logf(pixel);
    if (t_float < 0.0f) t_float = 0.0f;
    float t_spike = ceilf(t_float);
    if (t_spike > TIME_WINDOW) out_spikes[idx] = INF_TIME;
    else out_spikes[idx] = t_spike;
}

__global__ void k_layer_inference_thread_opt(
    const float* input_spikes, float* output_spikes,
    const float* kernel, const float* bias,
    const float* LUT_decay, const float* LUT_th,
    int t_min, int B,
    int H_in, int W_in, int C_in,
    int H_out, int W_out, int C_out,
    int K_h, int K_w, bool is_fc
) {
    int neuron_id = blockIdx.x * blockDim.x + threadIdx.x;
    int total_neurons = B * H_out * W_out * C_out;
    if (neuron_id >= total_neurons) return;

    int image_id = neuron_id / (H_out * W_out * C_out);
    int local_id = neuron_id % (H_out * W_out * C_out);

    int c_out, w_out = 0, h_out = 0;
    if (is_fc) {
        c_out = local_id;
    } else {
        c_out = local_id % C_out;
        w_out = (local_id / C_out) % W_out;
        h_out = local_id / (C_out * W_out);
    }

    float b_val = bias[c_out];
    float my_delta[80] = {0.0f}; 
    const int T = 80;

    // --- 第 1 步：生成输入矩阵 ---
    if (is_fc) {
        for (int i = 0; i < C_in; ++i) {
            float t_in = input_spikes[image_id * C_in + i];
            if (t_in < INF_TIME) {
                float arrival = t_in - TIME_FIRE_START;
                int t = (int)floorf(arrival);
                if (t < 0) t = 0; 
                if (t < T) {
                    int t_in_int = (int)t_in;
                    if (t_in_int < 0) t_in_int = 0; else if (t_in_int > T) t_in_int = T;
                    my_delta[t] += kernel[i * C_out + c_out] * LUT_decay[t_in_int];
                }
            }
        }
    } else {
        int pad_h = K_h / 2, pad_w = K_w / 2;
        int h_in_start = h_out - pad_h, w_in_start = w_out - pad_w;
        for (int kh = 0; kh < K_h; ++kh) {
            for (int kw = 0; kw < K_w; ++kw) {
                int h_in = h_in_start + kh, w_in = w_in_start + kw;
                if (h_in >= 0 && h_in < H_in && w_in >= 0 && w_in < W_in) {
                    for (int cin = 0; cin < C_in; ++cin) {
                        float t_in = input_spikes[image_id * (H_in * W_in * C_in) + (h_in * W_in + w_in) * C_in + cin];
                        if (t_in < INF_TIME) {
                            float arrival = t_in - TIME_FIRE_START;
                            int t = (int)floorf(arrival);
                            if (t < 0) t = 0;
                            if (t < T) {
                                int t_in_int = (int)t_in;
                                if (t_in_int < 0) t_in_int = 0; else if (t_in_int > T) t_in_int = T;
                                int w_idx = ((kh * K_w + kw) * C_in + cin) * C_out + c_out;
                                my_delta[t] += kernel[w_idx] * LUT_decay[t_in_int]; 
                            }
                        }
                    }
                }
            }
        }
    }

    // ✅ 第 2 步：论文核心 - 初始预测与掩码剔除机制
    float total_sum = 0.0f;
    for (int t = 0; t < T; ++t) total_sum += my_delta[t];
    float min_th = LUT_th[T - 1]; // 指数衰减阈值的最小值
    if (b_val + total_sum < min_th) {
        output_spikes[neuron_id] = PRUNED_TIME; // 标记为预测失败，跳过所有 Chunk 的积分
        return; 
    }

    // --- 第 3 步：独立早退积分器 ---
    float v_mem_acc = b_val;
    float final_spike = INF_TIME;

    for (int t = 0; t < T; ++t) {
        v_mem_acc += my_delta[t];
        float th = LUT_th[t];
        if (t >= t_min && v_mem_acc >= th && th >= 1e-5f) {
            final_spike = (float)t;
            break; // 🚀 触发硬件早退！
        }
    }
    output_spikes[neuron_id] = final_spike;
}

// VMEM 层 (最后一层无早退概念)
__global__ void k_layer_inference_vmem_optimized(
    const float* input_spikes, float* output_vmem,
    const float* kernel, const float* bias,
    const float* LUT_decay,
    int B, int C_in, int C_out
) {
    int local_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (local_id >= B * C_out) return;

    int image_id = local_id / C_out;
    int c_out = local_id % C_out;

    float b_val = bias[c_out];
    float final_v_mem_acc = 0.0f;
    const int T = (int)TIME_WINDOW;

    for (int i = 0; i < C_in; ++i) {
        float t_in = input_spikes[image_id * C_in + i];
        if (t_in < INF_TIME) {
            float arrival = t_in - TIME_FIRE_START;
            if (arrival <= T) { 
                int t_in_int = (int)t_in;
                if (t_in_int < 0) t_in_int = 0; else if (t_in_int > T) t_in_int = T; 
                final_v_mem_acc += kernel[i * C_out + c_out] * LUT_decay[t_in_int];
            }
        }
    }
    output_vmem[local_id] = final_v_mem_acc + b_val;
}

__global__ void k_pooling(const float* in_spikes, float* out_spikes, int B, int H_in, int W_in, int C) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int H_out = H_in / 2, W_out = W_in / 2;
    if (idx >= B * H_out * W_out * C) return;
    
    int b = idx / (H_out * W_out * C);
    int local_idx = idx % (H_out * W_out * C);
    int c = local_idx % C;
    int w = (local_idx / C) % W_out;
    int h = local_idx / C / W_out;
    
    float min_t = INF_TIME;
    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 2; ++j) {
            int cur_h = h * 2 + i, cur_w = w * 2 + j;
            if (cur_h < H_in && cur_w < W_in) {
                float t = in_spikes[b * (H_in * W_in * C) + (cur_h * W_in + cur_w) * C + c];
                if (t < min_t) min_t = t;
            }
        }
    }
    out_spikes[idx] = min_t;
}

// ✅ 新增：用于在旁路进行高速 Gamma 统计的共享内存规约内核
__global__ void k_measure_gamma(const float* spikes, unsigned long long* d_processed_chunks, unsigned long long* d_total_chunks, int size) {
    __shared__ unsigned long long shared_processed[256];
    __shared__ unsigned long long shared_total[256];
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;
    
    unsigned long long local_p = 0;
    unsigned long long local_t = 0;
    
    if (idx < size) {
        local_t = 3; // N_chunk = ceil(80/32) = 3
        float t = spikes[idx];
        if (t == PRUNED_TIME) {
            local_p = 0; // 成功被预测剔除，0 Chunk
        } else if (t >= TIME_WINDOW) {
            local_p = 3; // 熬到窗口结束也没发射，硬着头皮算了 3 个 Chunk
        } else {
            int t_int = (int)t;
            if (t_int < 0) t_int = 0;
            if (t_int > 79) t_int = 79;
            local_p = (t_int / 32) + 1; // 在第 1、2、3 个 Chunk 中提早退出了！
        }
    }
    
    shared_processed[tid] = local_p;
    shared_total[tid] = local_t;
    __syncthreads();
    
    // 线程块内高速规约
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_processed[tid] += shared_processed[tid + s];
            shared_total[tid] += shared_total[tid + s];
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        atomicAdd(d_processed_chunks, shared_processed[0]);
        atomicAdd(d_total_chunks, shared_total[0]);
    }
}

// ============== 辅助环境函数 ==============

void load_weights(const std::string& filename, std::vector<LayerWeights>& layers) {
    std::ifstream f(filename, std::ios::binary);
    if (!f.is_open()) { std::cerr << "File not found: " << filename << "\n"; exit(1); }
    int num_layers; f.read((char*)&num_layers, 4);
    for(int i=0; i<num_layers; ++i) {
        LayerWeights l; int ndim;
        f.read((char*)&ndim, 4); std::vector<int> k_shape(ndim); int k_size = 1;
        for(int j=0; j<ndim; ++j) { f.read((char*)&k_shape[j], 4); k_size *= k_shape[j]; }
        if (ndim == 4) { l.k_h = k_shape[0]; l.k_w = k_shape[1]; l.c_in = k_shape[2]; l.c_out = k_shape[3]; }
        else { l.k_h = 1; l.k_w = 1; l.c_in = k_shape[0]; l.c_out = k_shape[1]; }
        std::vector<float> h_kernel(k_size); f.read((char*)h_kernel.data(), k_size * 4);
        CHECK_CUDA(cudaMalloc(&l.d_kernel, k_size * 4)); CHECK_CUDA(cudaMemcpy(l.d_kernel, h_kernel.data(), k_size * 4, cudaMemcpyHostToDevice));
        f.read((char*)&ndim, 4); int b_size; f.read((char*)&b_size, 4);
        std::vector<float> h_bias(b_size); f.read((char*)h_bias.data(), b_size * 4);
        CHECK_CUDA(cudaMalloc(&l.d_bias, b_size * 4)); CHECK_CUDA(cudaMemcpy(l.d_bias, h_bias.data(), b_size * 4, cudaMemcpyHostToDevice));
        f.read((char*)&ndim, 4); int tc_size; f.read((char*)&tc_size, 4); f.read((char*)&l.tc_fire, 4);
        f.read((char*)&ndim, 4); int td_size; f.read((char*)&td_size, 4); f.read((char*)&l.td, 4);
        layers.push_back(l);
    }
}

void update_LUTs(float* d_LUT_decay, float* d_LUT_th, float tc_integ, float td_integ, float tc_fire, float td_fire) {
    const int T = (int)TIME_WINDOW;
    std::vector<float> h_LUT_decay(T + 1); 
    std::vector<float> h_LUT_th(T);
    for (int t = 0; t <= T; ++t) h_LUT_decay[t] = expf(-((float)t - td_integ) / tc_integ);
    for (int t = 0; t < T; ++t) h_LUT_th[t] = VTH_INIT * expf(-((float)t - td_fire) / tc_fire);
    cudaMemcpy(d_LUT_decay, h_LUT_decay.data(), (T + 1) * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(d_LUT_th, h_LUT_th.data(), T * 4, cudaMemcpyHostToDevice);
}

void measure_gamma(float* d_spikes, int num_neurons, unsigned long long* d_processed, unsigned long long* d_total) {
    int threads = 256;
    int blocks = (num_neurons + threads - 1) / threads;
    k_measure_gamma<<<blocks, threads>>>(d_spikes, d_processed, d_total, num_neurons);
}

void launch_conv_opt(float* d_in, float* d_out, int H_in, int W_in, int C_in, int H_out, int W_out, int C_out, int layer_idx, float prev_tc, float prev_td, float* d_LUT_decay, float* d_LUT_th, const std::vector<LayerWeights>& layers, int B, unsigned long long* d_p, unsigned long long* d_t) {
    int t_min = (int)ceilf(layers[layer_idx].td);
    if (t_min < 0) t_min = 0;
    
    int num_neurons = B * H_out * W_out * C_out;
    int threads = 256;
    int blocks = (num_neurons + threads - 1) / threads;
    
    k_layer_inference_thread_opt<<<blocks, threads>>>(
        d_in, d_out, layers[layer_idx].d_kernel, layers[layer_idx].d_bias, d_LUT_decay, d_LUT_th, t_min,
        B, H_in, W_in, C_in, H_out, W_out, C_out, layers[layer_idx].k_h, layers[layer_idx].k_w, false
    );
    // ✅ 测量早退因子（池化层不测，因为池化层没有积分器运算）
    measure_gamma(d_out, num_neurons, d_p, d_t);
}

void launch_fc_opt(float* d_in, float* d_out, int C_in, int C_out, int layer_idx, float prev_tc, float prev_td, float* d_LUT_decay, float* d_LUT_th, const std::vector<LayerWeights>& layers, int B, unsigned long long* d_p, unsigned long long* d_t) {
    int t_min = (int)ceilf(layers[layer_idx].td);
    if (t_min < 0) t_min = 0;
    
    int num_neurons = B * C_out;
    int threads = 256;
    int blocks = (num_neurons + threads - 1) / threads;

    k_layer_inference_thread_opt<<<blocks, threads>>>(
        d_in, d_out, layers[layer_idx].d_kernel, layers[layer_idx].d_bias, d_LUT_decay, d_LUT_th, t_min,
        B, 1, 1, C_in, 1, 1, C_out, 1, 1, true
    );
    measure_gamma(d_out, num_neurons, d_p, d_t);
}

void launch_fc_vmem_opt(float* d_in, float* d_out, int C_in, int C_out, int layer_idx, float prev_tc, float prev_td, float* d_LUT_decay, float* d_LUT_th, const std::vector<LayerWeights>& layers, int B) {
    int num_neurons = B * C_out;
    int threads = 256;
    int blocks = (num_neurons + threads - 1) / threads;

    k_layer_inference_vmem_optimized<<<blocks, threads>>>(
        d_in, d_out, layers[layer_idx].d_kernel, layers[layer_idx].d_bias, d_LUT_decay, B, C_in, C_out
    );
}

void launch_pool(float* d_in, float* d_out, int H_in, int W_in, int C, int B) {
    int num_neurons = B * (H_in / 2) * (W_in / 2) * C;
    int threads = 256;
    int blocks = (num_neurons + threads - 1) / threads;
    k_pooling<<<blocks, threads>>>(d_in, d_out, B, H_in, W_in, C);
}

// ================= 主程序 =================

int main() {
    const int TOTAL_IMAGES = 10000; 
    const int B = 1000; 
    const int NUM_BATCHES = TOTAL_IMAGES / B;
    const int T = (int)TIME_WINDOW;

    std::cout << ">>> Loading VGG16 Weights & Data..." << std::endl;
    std::vector<LayerWeights> layers;
    load_weights("exported_models/snn_weights_vgg.bin", layers);

    std::vector<float> all_imgs(TOTAL_IMAGES * 3072);
    for (int i = 0; i < TOTAL_IMAGES; ++i) {
        std::ifstream img_f("dataset_downloaded/cifar10_float/" + std::to_string(i) + ".bin", std::ios::binary);
        img_f.read((char*)&all_imgs[i * 3072], 3072 * 4);
    }
    std::vector<float> all_labels(TOTAL_IMAGES * 10);
    std::ifstream lbl_f("dataset_downloaded/cifar10_float/label_onehot", std::ios::binary);
    for (int i = 0; i < TOTAL_IMAGES; ++i) {
        lbl_f.seekg(i * 10 * 4, std::ios::beg);
        lbl_f.read((char*)&all_labels[i * 10], 10 * 4);
    }

    float *d_img, *d_s0, *d_c1, *d_c1_1, *d_p1, *d_c2, *d_c2_1, *d_p2;
    float *d_c3, *d_c3_1, *d_c3_2, *d_p3, *d_c4, *d_c4_1, *d_c4_2, *d_p4;
    float *d_c5, *d_c5_1, *d_c5_2, *d_p5, *d_fc1, *d_fc2, *d_fc3;
    
    CHECK_CUDA(cudaMalloc(&d_img, B * 3072 * 4)); CHECK_CUDA(cudaMalloc(&d_s0, B * 3072 * 4)); 
    CHECK_CUDA(cudaMalloc(&d_c1, B*32*32*64*4)); CHECK_CUDA(cudaMalloc(&d_c1_1, B*32*32*64*4)); CHECK_CUDA(cudaMalloc(&d_p1, B*16*16*64*4));
    CHECK_CUDA(cudaMalloc(&d_c2, B*16*16*128*4)); CHECK_CUDA(cudaMalloc(&d_c2_1, B*16*16*128*4)); CHECK_CUDA(cudaMalloc(&d_p2, B*8*8*128*4));
    CHECK_CUDA(cudaMalloc(&d_c3, B*8*8*256*4)); CHECK_CUDA(cudaMalloc(&d_c3_1, B*8*8*256*4)); CHECK_CUDA(cudaMalloc(&d_c3_2, B*8*8*256*4)); CHECK_CUDA(cudaMalloc(&d_p3, B*4*4*256*4));
    CHECK_CUDA(cudaMalloc(&d_c4, B*4*4*512*4)); CHECK_CUDA(cudaMalloc(&d_c4_1, B*4*4*512*4)); CHECK_CUDA(cudaMalloc(&d_c4_2, B*4*4*512*4)); CHECK_CUDA(cudaMalloc(&d_p4, B*2*2*512*4));
    CHECK_CUDA(cudaMalloc(&d_c5, B*2*2*512*4)); CHECK_CUDA(cudaMalloc(&d_c5_1, B*2*2*512*4)); CHECK_CUDA(cudaMalloc(&d_c5_2, B*2*2*512*4)); CHECK_CUDA(cudaMalloc(&d_p5, B*1*1*512*4));
    CHECK_CUDA(cudaMalloc(&d_fc1, B*512*4)); CHECK_CUDA(cudaMalloc(&d_fc2, B*512*4)); CHECK_CUDA(cudaMalloc(&d_fc3, B*10*4));

    float *d_LUT_decay, *d_LUT_th;
    CHECK_CUDA(cudaMalloc(&d_LUT_decay, (T + 1) * 4)); CHECK_CUDA(cudaMalloc(&d_LUT_th, T * 4));

    // ✅ 初始化全局测量变量
    unsigned long long *d_p_chunks, *d_t_chunks;
    CHECK_CUDA(cudaMalloc(&d_p_chunks, sizeof(unsigned long long)));
    CHECK_CUDA(cudaMalloc(&d_t_chunks, sizeof(unsigned long long)));
    CHECK_CUDA(cudaMemset(d_p_chunks, 0, sizeof(unsigned long long)));
    CHECK_CUDA(cudaMemset(d_t_chunks, 0, sizeof(unsigned long long)));

    std::cout << ">>> Starting Full 10K Inference with Gamma Measurement..." << std::endl;
    int total_correct = 0;
    float tc_in = 34.75016403198242f, td_in = 0.0f; 

    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);
    cudaEventRecord(start);

    for (int batch = 0; batch < NUM_BATCHES; ++batch) {
        CHECK_CUDA(cudaMemcpy(d_img, &all_imgs[batch * B * 3072], B * 3072 * 4, cudaMemcpyHostToDevice));

        k_encode_image<<<(B * 3072 + 255)/256, 256>>>(d_img, d_s0, B * 3072, tc_in, td_in);

        update_LUTs(d_LUT_decay, d_LUT_th, tc_in, td_in, layers[0].tc_fire, layers[0].td);
        launch_conv_opt(d_s0, d_c1, 32, 32, 3, 32, 32, 64, 0, tc_in, td_in, d_LUT_decay, d_LUT_th, layers, B, d_p_chunks, d_t_chunks);
        
        update_LUTs(d_LUT_decay, d_LUT_th, layers[0].tc_fire, layers[0].td, layers[1].tc_fire, layers[1].td);
        launch_conv_opt(d_c1, d_c1_1, 32, 32, 64, 32, 32, 64, 1, layers[0].tc_fire, layers[0].td, d_LUT_decay, d_LUT_th, layers, B, d_p_chunks, d_t_chunks);
        launch_pool(d_c1_1, d_p1, 32, 32, 64, B);

        update_LUTs(d_LUT_decay, d_LUT_th, layers[1].tc_fire, layers[1].td, layers[2].tc_fire, layers[2].td);
        launch_conv_opt(d_p1, d_c2, 16, 16, 64, 16, 16, 128, 2, layers[1].tc_fire, layers[1].td, d_LUT_decay, d_LUT_th, layers, B, d_p_chunks, d_t_chunks);
        
        update_LUTs(d_LUT_decay, d_LUT_th, layers[2].tc_fire, layers[2].td, layers[3].tc_fire, layers[3].td);
        launch_conv_opt(d_c2, d_c2_1, 16, 16, 128, 16, 16, 128, 3, layers[2].tc_fire, layers[2].td, d_LUT_decay, d_LUT_th, layers, B, d_p_chunks, d_t_chunks);
        launch_pool(d_c2_1, d_p2, 16, 16, 128, B);

        update_LUTs(d_LUT_decay, d_LUT_th, layers[3].tc_fire, layers[3].td, layers[4].tc_fire, layers[4].td);
        launch_conv_opt(d_p2, d_c3, 8, 8, 128, 8, 8, 256, 4, layers[3].tc_fire, layers[3].td, d_LUT_decay, d_LUT_th, layers, B, d_p_chunks, d_t_chunks);
        
        update_LUTs(d_LUT_decay, d_LUT_th, layers[4].tc_fire, layers[4].td, layers[5].tc_fire, layers[5].td);
        launch_conv_opt(d_c3, d_c3_1, 8, 8, 256, 8, 8, 256, 5, layers[4].tc_fire, layers[4].td, d_LUT_decay, d_LUT_th, layers, B, d_p_chunks, d_t_chunks);

        update_LUTs(d_LUT_decay, d_LUT_th, layers[5].tc_fire, layers[5].td, layers[6].tc_fire, layers[6].td);
        launch_conv_opt(d_c3_1, d_c3_2, 8, 8, 256, 8, 8, 256, 6, layers[5].tc_fire, layers[5].td, d_LUT_decay, d_LUT_th, layers, B, d_p_chunks, d_t_chunks);
        launch_pool(d_c3_2, d_p3, 8, 8, 256, B);

        update_LUTs(d_LUT_decay, d_LUT_th, layers[6].tc_fire, layers[6].td, layers[7].tc_fire, layers[7].td);
        launch_conv_opt(d_p3, d_c4, 4, 4, 256, 4, 4, 512, 7, layers[6].tc_fire, layers[6].td, d_LUT_decay, d_LUT_th, layers, B, d_p_chunks, d_t_chunks);

        update_LUTs(d_LUT_decay, d_LUT_th, layers[7].tc_fire, layers[7].td, layers[8].tc_fire, layers[8].td);
        launch_conv_opt(d_c4, d_c4_1, 4, 4, 512, 4, 4, 512, 8, layers[7].tc_fire, layers[7].td, d_LUT_decay, d_LUT_th, layers, B, d_p_chunks, d_t_chunks);

        update_LUTs(d_LUT_decay, d_LUT_th, layers[8].tc_fire, layers[8].td, layers[9].tc_fire, layers[9].td);
        launch_conv_opt(d_c4_1, d_c4_2, 4, 4, 512, 4, 4, 512, 9, layers[8].tc_fire, layers[8].td, d_LUT_decay, d_LUT_th, layers, B, d_p_chunks, d_t_chunks);
        launch_pool(d_c4_2, d_p4, 4, 4, 512, B);

        update_LUTs(d_LUT_decay, d_LUT_th, layers[9].tc_fire, layers[9].td, layers[10].tc_fire, layers[10].td);
        launch_conv_opt(d_p4, d_c5, 2, 2, 512, 2, 2, 512, 10, layers[9].tc_fire, layers[9].td, d_LUT_decay, d_LUT_th, layers, B, d_p_chunks, d_t_chunks);

        update_LUTs(d_LUT_decay, d_LUT_th, layers[10].tc_fire, layers[10].td, layers[11].tc_fire, layers[11].td);
        launch_conv_opt(d_c5, d_c5_1, 2, 2, 512, 2, 2, 512, 11, layers[10].tc_fire, layers[10].td, d_LUT_decay, d_LUT_th, layers, B, d_p_chunks, d_t_chunks);

        update_LUTs(d_LUT_decay, d_LUT_th, layers[11].tc_fire, layers[11].td, layers[12].tc_fire, layers[12].td);
        launch_conv_opt(d_c5_1, d_c5_2, 2, 2, 512, 2, 2, 512, 12, layers[11].tc_fire, layers[11].td, d_LUT_decay, d_LUT_th, layers, B, d_p_chunks, d_t_chunks);
        launch_pool(d_c5_2, d_p5, 2, 2, 512, B);

        update_LUTs(d_LUT_decay, d_LUT_th, layers[12].tc_fire, layers[12].td, layers[13].tc_fire, layers[13].td);
        launch_fc_opt(d_p5, d_fc1, 512, 512, 13, layers[12].tc_fire, layers[12].td, d_LUT_decay, d_LUT_th, layers, B, d_p_chunks, d_t_chunks);

        update_LUTs(d_LUT_decay, d_LUT_th, layers[13].tc_fire, layers[13].td, layers[14].tc_fire, layers[14].td);
        launch_fc_opt(d_fc1, d_fc2, 512, 512, 14, layers[13].tc_fire, layers[13].td, d_LUT_decay, d_LUT_th, layers, B, d_p_chunks, d_t_chunks);

        update_LUTs(d_LUT_decay, d_LUT_th, layers[14].tc_fire, layers[14].td, layers[15].tc_fire, layers[15].td);
        launch_fc_vmem_opt(d_fc2, d_fc3, 512, 10, 15, layers[14].tc_fire, layers[14].td, d_LUT_decay, d_LUT_th, layers, B);

        std::vector<float> h_out(B * 10);
        CHECK_CUDA(cudaMemcpy(h_out.data(), d_fc3, B * 10 * 4, cudaMemcpyDeviceToHost));

        int batch_correct = 0;
        for (int bi = 0; bi < B; ++bi) {
            float max_v = -1e9f; int pred_cls = -1;
            for (int i = 0; i < 10; ++i) {
                if (h_out[bi * 10 + i] > max_v) { max_v = h_out[bi * 10 + i]; pred_cls = i; }
            }
            int true_cls = -1;
            for (int i = 0; i < 10; ++i) {
                if (all_labels[(batch * B + bi) * 10 + i] > 0.5f) true_cls = i;
            }
            if (pred_cls == true_cls) ++batch_correct;
        }
        total_correct += batch_correct;
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    // ✅ 获取并计算最终的早退因子 Gamma
    unsigned long long final_processed = 0, final_total = 0;
    CHECK_CUDA(cudaMemcpy(&final_processed, d_p_chunks, sizeof(unsigned long long), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(&final_total, d_t_chunks, sizeof(unsigned long long), cudaMemcpyDeviceToHost));
    
    double gamma = (double)final_processed / (double)final_total;

    float final_accuracy = static_cast<float>(total_correct) / TOTAL_IMAGES * 100.0f;
    std::cout << "\n=======================================" << std::endl;
    std::cout << "🚀 FULL CIFAR-10 SNN MEASUREMENT COMPLETED!" << std::endl;
    std::cout << "Total Images:           " << TOTAL_IMAGES << std::endl;
    std::cout << "Final Accuracy:         " << final_accuracy << "%" << std::endl;
    std::cout << "Total Inference Time:   " << milliseconds / 1000.0f << " seconds" << std::endl;
    std::cout << "Throughput:             " << (TOTAL_IMAGES / (milliseconds / 1000.0f)) << " images/sec" << std::endl;
    std::cout << "---------------------------------------" << std::endl;
    std::cout << "🎯 Early Exit Factor (Gamma): " << gamma << std::endl;
    std::cout << "   - Processed Chunks:  " << final_processed << std::endl;
    std::cout << "   - Total Poss. Chunks:" << final_total << std::endl;
    std::cout << "=======================================" << std::endl;

    return 0;
}
