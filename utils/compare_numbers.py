def compare_numbers(a, b, c, d, compare_number):
    numbers = [a, b, c, d]
    count = sum(1 for number in numbers if compare_number < number)
    return count

def calculate_difference(a, b, c, d, operation, compare_number):
    numbers = [a, b, c, d]

    if operation == 1:
        # 和最小值的差值
        min_value = min(numbers)
        return  min_value-compare_number
    elif operation == 2:
        # 和最大值的差值
        max_value = max(numbers)
        return  max_value-compare_number
    elif operation == 3:

                # 和最大值的差值

            max_value = min(numbers)

            return  max_value-compare_number
    else:
        # 非法的运算标识符
        return "Invalid operation"


