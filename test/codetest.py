#冒泡排序
def bubble_sort(arr):
    for i in range(len(arr)-1):
        for j in range(len(arr)-1-i):
            if arr[j]>arr[j+1]:
                arr[j],arr[j+1]=arr[j+1],arr[j]
    return arr
arr=[100,1,2,3,4,5,6,7,8,9,10]
print(bubble_sort(arr))
#快速排序

#从大到小排序
def quick_sort(arr):
    if len(arr)<=1:
        return arr
    else:
        pivot=arr[0]
        less=[i for i in arr[1:] if i<=pivot]
        greater=[i for i in arr[1:] if i>pivot]
        return quick_sort(less)+[pivot]+quick_sort(greater)


arr=[6.5,1,2,3,4,5,6,7,8,9,10]
print(quick_sort(arr))