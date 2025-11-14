import os


def find_order_excel_files(directory):
    result = []
    for filename in os.listdir(directory):
        if filename.endswith('.xlsx') and '订单' in filename:
            result.append(os.path.join(directory, filename))
    return result


directory_path = r'/'
matched_files = find_order_excel_files(directory_path)

print("匹配到的文件列表：")
for file in matched_files:
    print(file)
