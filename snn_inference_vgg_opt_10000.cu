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

// 核心优化修复版
__global__ void k_layer_inference_optimized(
    const float* input_spikes, float* output_spikes,
    const float* kernel, const float* bias,
    const float* LUT_decay, const float* LUT_th,
    int t_min, int B,
    int H_in, int W_in, int C_in,
    int H_out, int W_out, int C_out,
    int K_h, int K_w, bool is_fc
) {
    const int T = (int)TIME_WINDOW;
    const int WARP_SIZE = 32;
    int warps_per_block = blockDim.x / WARP_SIZE;
    int neuron_id = blockIdx.x * warps_per_block + (threadIdx.x / WARP_SIZE);
    
    int num_neurons_per_image = H_out * W_out * C_out;
    int total_neurons = B * num_neurons_per_image;
    if (neuron_id >= total_neurons) return;

    int lane = threadIdx.x % WARP_SIZE;
    int image_id = neuron_id / num_neurons_per_image;
    int local_id = neuron_id % num_neurons_per_image;
    
    int c_out, w_out = 0, h_out = 0;
    if (is_fc) {
        c_out = local_id;
    } else {
        c_out = local_id % C_out;
        w_out = (local_id / C_out) % W_out;
        h_out = local_id / C_out / W_out;
    }
    
    float b_val = bias[c_out];
    extern __shared__ float shared_delta[];
    float* my_delta = shared_delta + (threadIdx.x / WARP_SIZE) * T;
    
    // 初始化 delta 数组
    for (int i = lane; i < T; i += WARP_SIZE) my_delta[i] = 0.0f;
    __syncwarp();

    // 🚀 性能优化：全 Warp 32线程并行处理输入突触计算，不再是 lane==0 单干
    if (is_fc) {
        for (int i = lane; i < C_in; i += WARP_SIZE) {
            float t_in = input_spikes[image_id * C_in + i];
            if (t_in < INF_TIME) {
                float arrival_time = t_in - TIME_FIRE_START; 
                int t = (int)floorf(arrival_time);
                if (t < 0) t = 0; 
                if (t < T) {
                    int t_in_int = (int)t_in;
                    if (t_in_int < 0) t_in_int = 0;
                    if (t_in_int >= T + 1) t_in_int = T; 
                    float decay = LUT_decay[t_in_int]; 
                    atomicAdd(&my_delta[t], kernel[i * C_out + c_out] * decay);
                }
            }
        }
    } else {
        int pad_h = K_h / 2, pad_w = K_w / 2;
        int h_in_start = h_out - pad_h, w_in_start = w_out - pad_w;
        int total_inputs = K_h * K_w * C_in;
        
        for (int i = lane; i < total_inputs; i += WARP_SIZE) {
            int cin = i % C_in;
            int kw = (i / C_in) % K_w;
            int kh = i / (C_in * K_w);
            int h_in = h_in_start + kh;
            int w_in = w_in_start + kw;
            
            if (h_in >= 0 && h_in < H_in && w_in >= 0 && w_in < W_in) {
                float t_in = input_spikes[image_id * (H_in * W_in * C_in) + (h_in * W_in + w_in) * C_in + cin];
                if (t_in < INF_TIME) {
                    float arrival_time = t_in - TIME_FIRE_START;
                    int t = (int)floorf(arrival_time);
                    if (t < 0) t = 0;
                    if (t < T) {
                        int t_in_int = (int)t_in;
                        if (t_in_int < 0) t_in_int = 0;
                        if (t_in_int >= T + 1) t_in_int = T; 
                        float decay = LUT_decay[t_in_int];
                        int w_idx = ((kh * K_w + kw) * C_in + cin) * C_out + c_out;
                        atomicAdd(&my_delta[t], kernel[w_idx] * decay);
                    }
                }
            }
        }
    }
    __syncwarp(); // 确保所有 atomicAdd 完成

    // 预测神经元是否可能发放
    float total_sum = 0.0f;
    for (int base = 0; base < T; base += WARP_SIZE) {
        float val = (base + lane < T) ? my_delta[base + lane] : 0.0f;
        for (int offset = 16; offset > 0; offset /= 2) val += __shfl_down_sync(0xffffffffu, val, offset);
        if (lane == 0) total_sum += val;
    }
    __syncwarp();
    
    total_sum = __shfl_sync(0xffffffffu, total_sum, 0); 
    float min_th = LUT_th[T - 1]; 
    if (b_val + total_sum < min_th) {
        if (lane == 0) output_spikes[neuron_id] = INF_TIME;
        return; 
    }

    // 并行前缀和与协同早退
    float current_base = 0.0f;
    const int chunk_size = WARP_SIZE;
    int num_chunks = (T + chunk_size - 1) / chunk_size;
    
    for (int ch = 0; ch < num_chunks; ++ch) {
        int t_start = ch * chunk_size;
        float val = (t_start + lane < T) ? my_delta[t_start + lane] : 0.0f;
        float prefix = val;
        
        for (int d = 1; d < chunk_size; d *= 2) {
            float up = __shfl_up_sync(0xffffffffu, prefix, d);
            if (lane >= d) prefix += up;
        }
        
        float full_prefix = current_base + prefix;
        float th = (t_start + lane < T) ? LUT_th[t_start + lane] : 0.0f;
        
        bool pred = (t_start + lane < T) && (t_start + lane >= t_min) && (full_prefix + b_val >= th) && (th >= 1e-5f);
        unsigned int mask = __ballot_sync(0xffffffffu, pred);
        
        if (mask != 0) { 
            int first = __ffs(mask) - 1;
            if (lane == 0) output_spikes[neuron_id] = (float)(t_start + first);
            return;
        }
        
        float block_sum = __shfl_sync(0xffffffffu, prefix, chunk_size - 1);
        
        // 🐛 漏洞修复点：去掉了 if (lane == 0)，强制要求 Warp 内所有线程一同更新历史积分基数
        current_base += block_sum; 
        
        __syncwarp();
        
        float remaining_sum = total_sum - current_base;
        if (current_base + b_val + remaining_sum < min_th) { 
            if (lane == 0) output_spikes[neuron_id] = INF_TIME;
            return;
        }
    }
    if (lane == 0) output_spikes[neuron_id] = INF_TIME;
}

__global__ void k_layer_inference_vmem_optimized(
    const float* input_spikes, float* output_vmem,
    const float* kernel, const float* bias,
    const float* LUT_decay,
    int B, int C_in, int C_out
) {
    int local_id = blockIdx.x * blockDim.x + threadIdx.x;
    int total_neurons = B * C_out;
    if (local_id >= total_neurons) return;

    int image_id = local_id / C_out;
    int c_out = local_id % C_out;

    float b_val = bias[c_out];
    float final_v_mem_acc = 0.0f;
    const int T = (int)TIME_WINDOW;

    for (int i = 0; i < C_in; ++i) {
        float t_in = input_spikes[image_id * C_in + i];
        if (t_in < INF_TIME) {
            float arrival_time = t_in - TIME_FIRE_START;
            if (arrival_time <= T) { 
                int t_in_int = (int)t_in;
                if (t_in_int < 0) t_in_int = 0;
                if (t_in_int >= T + 1) t_in_int = T; 
                final_v_mem_acc += kernel[i * C_out + c_out] * LUT_decay[t_in_int];
            }
        }
    }
    output_vmem[local_id] = final_v_mem_acc + b_val;
}

__global__ void k_pooling(const float* in_spikes, float* out_spikes, int B, int H_in, int W_in, int C) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int H_out = H_in / 2, W_out = W_in / 2;
    int total = B * H_out * W_out * C;
    if (idx >= total) return;
    
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

void launch_conv_opt(float* d_in, float* d_out, int H_in, int W_in, int C_in, int H_out, int W_out, int C_out, int layer_idx, float prev_tc, float prev_td, float* d_LUT_decay, float* d_LUT_th, const std::vector<LayerWeights>& layers, int B) {
    int t_min = (int)ceilf(layers[layer_idx].td);
    if (t_min < 0) t_min = 0;
    int num_neurons = B * H_out * W_out * C_out;
    int BLOCK_SIZE = 128;
    int warps_per_block = BLOCK_SIZE / 32;
    int num_blocks = (num_neurons + warps_per_block - 1) / warps_per_block;
    
    k_layer_inference_optimized<<<num_blocks, BLOCK_SIZE, warps_per_block * 80 * sizeof(float)>>>(
        d_in, d_out, layers[layer_idx].d_kernel, layers[layer_idx].d_bias, d_LUT_decay, d_LUT_th, t_min,
        B, H_in, W_in, C_in, H_out, W_out, C_out, layers[layer_idx].k_h, layers[layer_idx].k_w, false
    );
    CHECK_CUDA(cudaDeviceSynchronize());
}

void launch_fc_opt(float* d_in, float* d_out, int C_in, int C_out, int layer_idx, float prev_tc, float prev_td, float* d_LUT_decay, float* d_LUT_th, const std::vector<LayerWeights>& layers, int B) {
    int t_min = (int)ceilf(layers[layer_idx].td);
    if (t_min < 0) t_min = 0;
    int num_neurons = B * C_out;
    int BLOCK_SIZE = 128;
    int warps_per_block = BLOCK_SIZE / 32;
    int num_blocks = (num_neurons + warps_per_block - 1) / warps_per_block;

    k_layer_inference_optimized<<<num_blocks, BLOCK_SIZE, warps_per_block * 80 * sizeof(float)>>>(
        d_in, d_out, layers[layer_idx].d_kernel, layers[layer_idx].d_bias, d_LUT_decay, d_LUT_th, t_min,
        B, 1, 1, C_in, 1, 1, C_out, 1, 1, true
    );
    CHECK_CUDA(cudaDeviceSynchronize());
}

void launch_fc_vmem_opt(float* d_in, float* d_out, int C_in, int C_out, int layer_idx, float prev_tc, float prev_td, float* d_LUT_decay, float* d_LUT_th, const std::vector<LayerWeights>& layers, int B) {
    int num_neurons = B * C_out;
    int threads = 256;
    int blocks = (num_neurons + threads - 1) / threads;

    k_layer_inference_vmem_optimized<<<blocks, threads>>>(
        d_in, d_out, layers[layer_idx].d_kernel, layers[layer_idx].d_bias, d_LUT_decay, B, C_in, C_out
    );
    CHECK_CUDA(cudaDeviceSynchronize());
}

void launch_pool(float* d_in, float* d_out, int H_in, int W_in, int C, int B) {
    int num_neurons = B * (H_in / 2) * (W_in / 2) * C;
    int threads = 256;
    int blocks = (num_neurons + threads - 1) / threads;
    k_pooling<<<blocks, threads>>>(d_in, d_out, B, H_in, W_in, C);
    CHECK_CUDA(cudaDeviceSynchronize());
}

int main() {
    const int TOTAL_IMAGES = 10000; // 测试集总数
    const int B = 1000;             // 单次送入 GPU 的 Batch Size
    const int NUM_BATCHES = TOTAL_IMAGES / B;
    const int T = (int)TIME_WINDOW;

    std::cout << ">>> Loading VGG16 Weights..." << std::endl;
    std::vector<LayerWeights> layers;
    load_weights("exported_models/snn_weights_vgg.bin", layers);

    std::cout << ">>> Loading All " << TOTAL_IMAGES << " CIFAR-10 Test Images..." << std::endl;
    std::vector<float> all_imgs(TOTAL_IMAGES * 3072);
    for (int i = 0; i < TOTAL_IMAGES; ++i) {
        std::ifstream img_f("dataset_downloaded/cifar10_float/" + std::to_string(i) + ".bin", std::ios::binary);
        if (!img_f.is_open()) {
            std::cerr << "Cannot open image " << i << ".bin" << std::endl;
            return 1;
        }
        img_f.read((char*)&all_imgs[i * 3072], 3072 * 4);
    }

    std::vector<float> all_labels(TOTAL_IMAGES * 10);
    std::ifstream lbl_f("dataset_downloaded/cifar10_float/label_onehot", std::ios::binary);
    for (int i = 0; i < TOTAL_IMAGES; ++i) {
        lbl_f.seekg(i * 10 * 4, std::ios::beg);
        lbl_f.read((char*)&all_labels[i * 10], 10 * 4);
    }

    // 只为 B=1000 分配显存，复用这块空间！
    float *d_img, *d_s0, *d_c1, *d_c1_1, *d_p1, *d_c2, *d_c2_1, *d_p2;
    float *d_c3, *d_c3_1, *d_c3_2, *d_p3, *d_c4, *d_c4_1, *d_c4_2, *d_p4;
    float *d_c5, *d_c5_1, *d_c5_2, *d_p5, *d_fc1, *d_fc2, *d_fc3;
    
    CHECK_CUDA(cudaMalloc(&d_img, B * 3072 * 4)); 
    CHECK_CUDA(cudaMalloc(&d_s0, B * 3072 * 4)); 
    CHECK_CUDA(cudaMalloc(&d_c1, B*32*32*64*4)); CHECK_CUDA(cudaMalloc(&d_c1_1, B*32*32*64*4)); CHECK_CUDA(cudaMalloc(&d_p1, B*16*16*64*4));
    CHECK_CUDA(cudaMalloc(&d_c2, B*16*16*128*4)); CHECK_CUDA(cudaMalloc(&d_c2_1, B*16*16*128*4)); CHECK_CUDA(cudaMalloc(&d_p2, B*8*8*128*4));
    CHECK_CUDA(cudaMalloc(&d_c3, B*8*8*256*4)); CHECK_CUDA(cudaMalloc(&d_c3_1, B*8*8*256*4)); CHECK_CUDA(cudaMalloc(&d_c3_2, B*8*8*256*4)); CHECK_CUDA(cudaMalloc(&d_p3, B*4*4*256*4));
    CHECK_CUDA(cudaMalloc(&d_c4, B*4*4*512*4)); CHECK_CUDA(cudaMalloc(&d_c4_1, B*4*4*512*4)); CHECK_CUDA(cudaMalloc(&d_c4_2, B*4*4*512*4)); CHECK_CUDA(cudaMalloc(&d_p4, B*2*2*512*4));
    CHECK_CUDA(cudaMalloc(&d_c5, B*2*2*512*4)); CHECK_CUDA(cudaMalloc(&d_c5_1, B*2*2*512*4)); CHECK_CUDA(cudaMalloc(&d_c5_2, B*2*2*512*4)); CHECK_CUDA(cudaMalloc(&d_p5, B*1*1*512*4));
    CHECK_CUDA(cudaMalloc(&d_fc1, B*512*4)); CHECK_CUDA(cudaMalloc(&d_fc2, B*512*4)); CHECK_CUDA(cudaMalloc(&d_fc3, B*10*4));

    float *d_LUT_decay, *d_LUT_th;
    CHECK_CUDA(cudaMalloc(&d_LUT_decay, (T + 1) * 4));
    CHECK_CUDA(cudaMalloc(&d_LUT_th, T * 4));

    std::cout << ">>> Starting Full 10K Dataset Inference (" << NUM_BATCHES << " Batches of " << B << ")..." << std::endl;
    
    int total_correct = 0;
    float tc_in = 34.75016403198242f, td_in = 0.0f; 
    int encode_threads = 256;

    // 创建 CUDA Events 用于精确测速
    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);
    cudaEventRecord(start);

    for (int batch = 0; batch < NUM_BATCHES; ++batch) {
        // 1. 将当前 Batch 的图片从主内存拷入显存
        CHECK_CUDA(cudaMemcpy(d_img, &all_imgs[batch * B * 3072], B * 3072 * 4, cudaMemcpyHostToDevice));

        // 2. 图像编码
        k_encode_image<<<(B * 3072 + encode_threads - 1)/encode_threads, encode_threads>>>(d_img, d_s0, B * 3072, tc_in, td_in); CHECK_CUDA(cudaDeviceSynchronize());

        // --- Block 1 ---
        update_LUTs(d_LUT_decay, d_LUT_th, tc_in, td_in, layers[0].tc_fire, layers[0].td);
        launch_conv_opt(d_s0, d_c1, 32, 32, 3, 32, 32, 64, 0, tc_in, td_in, d_LUT_decay, d_LUT_th, layers, B);
        
        update_LUTs(d_LUT_decay, d_LUT_th, layers[0].tc_fire, layers[0].td, layers[1].tc_fire, layers[1].td);
        launch_conv_opt(d_c1, d_c1_1, 32, 32, 64, 32, 32, 64, 1, layers[0].tc_fire, layers[0].td, d_LUT_decay, d_LUT_th, layers, B);
        launch_pool(d_c1_1, d_p1, 32, 32, 64, B);

        // --- Block 2 ---
        update_LUTs(d_LUT_decay, d_LUT_th, layers[1].tc_fire, layers[1].td, layers[2].tc_fire, layers[2].td);
        launch_conv_opt(d_p1, d_c2, 16, 16, 64, 16, 16, 128, 2, layers[1].tc_fire, layers[1].td, d_LUT_decay, d_LUT_th, layers, B);
        
        update_LUTs(d_LUT_decay, d_LUT_th, layers[2].tc_fire, layers[2].td, layers[3].tc_fire, layers[3].td);
        launch_conv_opt(d_c2, d_c2_1, 16, 16, 128, 16, 16, 128, 3, layers[2].tc_fire, layers[2].td, d_LUT_decay, d_LUT_th, layers, B);
        launch_pool(d_c2_1, d_p2, 16, 16, 128, B);

        // --- Block 3 ---
        update_LUTs(d_LUT_decay, d_LUT_th, layers[3].tc_fire, layers[3].td, layers[4].tc_fire, layers[4].td);
        launch_conv_opt(d_p2, d_c3, 8, 8, 128, 8, 8, 256, 4, layers[3].tc_fire, layers[3].td, d_LUT_decay, d_LUT_th, layers, B);
        
        update_LUTs(d_LUT_decay, d_LUT_th, layers[4].tc_fire, layers[4].td, layers[5].tc_fire, layers[5].td);
        launch_conv_opt(d_c3, d_c3_1, 8, 8, 256, 8, 8, 256, 5, layers[4].tc_fire, layers[4].td, d_LUT_decay, d_LUT_th, layers, B);

        update_LUTs(d_LUT_decay, d_LUT_th, layers[5].tc_fire, layers[5].td, layers[6].tc_fire, layers[6].td);
        launch_conv_opt(d_c3_1, d_c3_2, 8, 8, 256, 8, 8, 256, 6, layers[5].tc_fire, layers[5].td, d_LUT_decay, d_LUT_th, layers, B);
        launch_pool(d_c3_2, d_p3, 8, 8, 256, B);

        // --- Block 4 ---
        update_LUTs(d_LUT_decay, d_LUT_th, layers[6].tc_fire, layers[6].td, layers[7].tc_fire, layers[7].td);
        launch_conv_opt(d_p3, d_c4, 4, 4, 256, 4, 4, 512, 7, layers[6].tc_fire, layers[6].td, d_LUT_decay, d_LUT_th, layers, B);

        update_LUTs(d_LUT_decay, d_LUT_th, layers[7].tc_fire, layers[7].td, layers[8].tc_fire, layers[8].td);
        launch_conv_opt(d_c4, d_c4_1, 4, 4, 512, 4, 4, 512, 8, layers[7].tc_fire, layers[7].td, d_LUT_decay, d_LUT_th, layers, B);

        update_LUTs(d_LUT_decay, d_LUT_th, layers[8].tc_fire, layers[8].td, layers[9].tc_fire, layers[9].td);
        launch_conv_opt(d_c4_1, d_c4_2, 4, 4, 512, 4, 4, 512, 9, layers[8].tc_fire, layers[8].td, d_LUT_decay, d_LUT_th, layers, B);
        launch_pool(d_c4_2, d_p4, 4, 4, 512, B);

        // --- Block 5 ---
        update_LUTs(d_LUT_decay, d_LUT_th, layers[9].tc_fire, layers[9].td, layers[10].tc_fire, layers[10].td);
        launch_conv_opt(d_p4, d_c5, 2, 2, 512, 2, 2, 512, 10, layers[9].tc_fire, layers[9].td, d_LUT_decay, d_LUT_th, layers, B);

        update_LUTs(d_LUT_decay, d_LUT_th, layers[10].tc_fire, layers[10].td, layers[11].tc_fire, layers[11].td);
        launch_conv_opt(d_c5, d_c5_1, 2, 2, 512, 2, 2, 512, 11, layers[10].tc_fire, layers[10].td, d_LUT_decay, d_LUT_th, layers, B);

        update_LUTs(d_LUT_decay, d_LUT_th, layers[11].tc_fire, layers[11].td, layers[12].tc_fire, layers[12].td);
        launch_conv_opt(d_c5_1, d_c5_2, 2, 2, 512, 2, 2, 512, 12, layers[11].tc_fire, layers[11].td, d_LUT_decay, d_LUT_th, layers, B);
        launch_pool(d_c5_2, d_p5, 2, 2, 512, B);

        // --- FC Block ---
        update_LUTs(d_LUT_decay, d_LUT_th, layers[12].tc_fire, layers[12].td, layers[13].tc_fire, layers[13].td);
        launch_fc_opt(d_p5, d_fc1, 512, 512, 13, layers[12].tc_fire, layers[12].td, d_LUT_decay, d_LUT_th, layers, B);

        update_LUTs(d_LUT_decay, d_LUT_th, layers[13].tc_fire, layers[13].td, layers[14].tc_fire, layers[14].td);
        launch_fc_opt(d_fc1, d_fc2, 512, 512, 14, layers[13].tc_fire, layers[13].td, d_LUT_decay, d_LUT_th, layers, B);

        // --- Final VMEM Output ---
        update_LUTs(d_LUT_decay, d_LUT_th, layers[14].tc_fire, layers[14].td, layers[15].tc_fire, layers[15].td);
        launch_fc_vmem_opt(d_fc2, d_fc3, 512, 10, 15, layers[14].tc_fire, layers[14].td, d_LUT_decay, d_LUT_th, layers, B);

        // 3. 取回并验证结果
        std::vector<float> h_out(B * 10);
        CHECK_CUDA(cudaMemcpy(h_out.data(), d_fc3, B * 10 * 4, cudaMemcpyDeviceToHost));

        int batch_correct = 0;
        for (int bi = 0; bi < B; ++bi) {
            float max_v = -1e9f;
            int pred_cls = -1;
            for (int i = 0; i < 10; ++i) {
                float v = h_out[bi * 10 + i];
                if (v > max_v) { max_v = v; pred_cls = i; }
            }
            int true_cls = -1;
            int global_idx = batch * B + bi;
            for (int i = 0; i < 10; ++i) {
                if (all_labels[global_idx * 10 + i] > 0.5f) true_cls = i;
            }
            if (pred_cls == true_cls) ++batch_correct;
        }
        total_correct += batch_correct;
        
        std::cout << "  -> Batch [" << batch + 1 << "/" << NUM_BATCHES 
                  << "] completed. Correct: " << batch_correct << " / " << B << std::endl;
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    float final_accuracy = static_cast<float>(total_correct) / TOTAL_IMAGES * 100.0f;
    std::cout << "\n=======================================" << std::endl;
    std::cout << "🚀 FULL CIFAR-10 TEST COMPLETED!" << std::endl;
    std::cout << "Total Images Processed: " << TOTAL_IMAGES << std::endl;
    std::cout << "Final Accuracy:         " << final_accuracy << "%" << std::endl;
    std::cout << "Total Inference Time:   " << milliseconds / 1000.0f << " seconds" << std::endl;
    std::cout << "Throughput:             " << (TOTAL_IMAGES / (milliseconds / 1000.0f)) << " images/sec" << std::endl;
    std::cout << "=======================================" << std::endl;

    return 0;
}
