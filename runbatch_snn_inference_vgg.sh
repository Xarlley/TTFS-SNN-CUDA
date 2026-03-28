#!/bin/bash

# 检查可执行文件是否存在
if [ ! -f "./snn_inference_vgg" ]; then
    echo "错误: 未找到可执行文件 ./snn_inference_vgg。请先使用 nvcc 编译代码。"
    exit 1
fi

# 设置测试范围
START_IDX=0
END_IDX=100
TOTAL_RUNS=$((END_IDX - START_IDX + 1))
CORRECT_COUNT=0

echo "=================================================="
echo "开始批量 SNN 推理测试 (索引: $START_IDX 到 $END_IDX)"
echo "=================================================="

# 循环运行每个图片的推理
for i in $(seq $START_IDX $END_IDX); do
    # 运行程序并将标准输出和错误输出都捕获到变量中
    OUTPUT=$(./snn_inference_vgg $i 2>&1)
    
    # 检查输出文本中是否包含表示成功的关键字
    if echo "$OUTPUT" | grep -q "SUCCESS!"; then
        ((CORRECT_COUNT++))
        # 提取真实标签和预测标签用于显示（可选）
        pred=$(echo "$OUTPUT" | grep "Prediction:" | awk '{print $2}')
        gt=$(echo "$OUTPUT" | grep "Ground Truth:" | awk '{print $3}')
        echo "[Image $i] 成功! 预测: $pred, 真实: $gt"
    else
        pred=$(echo "$OUTPUT" | grep "Prediction:" | awk '{print $2}')
        gt=$(echo "$OUTPUT" | grep "Ground Truth:" | awk '{print $3}')
        echo "[Image $i] 失败. 预测: $pred, 真实: $gt"
    fi
done

echo "=================================================="
echo "测试完成！"
echo "总运行次数: $TOTAL_RUNS"
echo "正确预测数: $CORRECT_COUNT"

# 计算准确率并保留两位小数 (使用 bc 进行浮点数运算)
if command -v bc &> /dev/null; then
    ACCURACY=$(echo "scale=2; $CORRECT_COUNT / $TOTAL_RUNS * 100" | bc)
    echo "最终准确率: ${ACCURACY}%"
else
    # 如果系统没有安装 bc，使用 awk 计算
    ACCURACY=$(awk "BEGIN {printf \"%.2f\", ($CORRECT_COUNT / $TOTAL_RUNS) * 100}")
    echo "最终准确率: ${ACCURACY}%"
fi
echo "=================================================="
