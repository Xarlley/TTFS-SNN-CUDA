#!/bin/bash

# 定义日志文件
log_file="inference_log.txt"

# 初始化日志文件（清空现有内容，如果需要追加到旧日志，注释掉这一行）
> "$log_file"

# 初始化计数器
failed_count=0
success_count=0
unknown_count=0

# 循环从 0 到 30
for i in {0..4999}; do
    #echo "Running ./snn_inference5 $i..."  # 进度提示，便于跟踪

    # 记录开始分隔符和时间到日志
    echo "----- Run for $i -----" >> "$log_file"
    echo "Start time: $(date '+%Y-%m-%d %H:%M:%S')" >> "$log_file"
    echo "" >> "$log_file"  # 空行分隔

    # 捕获程序输出
    output=$(./snn_inference5 $i 2>&1)  # 捕获 stdout 和 stderr

    # 如果输出为空，记录提示
    if [ -z "$output" ]; then
        echo "No output from this run." >> "$log_file"
    else
        echo "$output" >> "$log_file"
    fi

    # 添加结束标记
    echo "" >> "$log_file"  # 空行
    echo "End time: $(date '+%Y-%m-%d %H:%M:%S')" >> "$log_file"
    echo "---------------------" >> "$log_file"
    echo "" >> "$log_file"  # 额外空行分隔下一个运行

    # 获取最后一行，并去除前后空白
    last_line=$(echo "$output" | tail -n 1 | sed 's/^[ \t]*//;s/[ \t]*$//')

    # 检查最后一行并记录结果到日志
    if [ "$last_line" = "FAILED." ]; then
        ((failed_count++))
        echo "Result: FAILED." >> "$log_file"
    elif [ "$last_line" = "SUCCESS!" ]; then
        ((success_count++))
        echo "Result: SUCCESS!" >> "$log_file"
    else
        ((unknown_count++))
        echo "Warning: Unexpected last line for $i: '$last_line'"  # 终端警告
        echo "Result: Unknown (last line: '$last_line')" >> "$log_file"
    fi
done

# 输出统计结果到终端
echo "-------------------"
echo "Statistics:"
echo "SUCCESS!: $success_count"
echo "FAILED.: $failed_count"
echo "Unknown: $unknown_count"
echo "Total runs: $((success_count + failed_count + unknown_count))"
echo "-------------------"
echo "All outputs have been saved to '$log_file'. You can view it with 'cat $log_file' or a text editor."