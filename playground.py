

def movezeros(nums):
    p = 0
    for i in range(len(nums)):
        current = nums[i]
        if current != 0:
            swap(i, p, nums)
            p += 1
    return nums


def swap(i, p, nums):
    temp = nums[i]
    nums[i] = nums[p]
    nums[p] = temp



def main():
    nums = [1, 0, 3, 0, 4, 5, 6]
    print(nums)
    print(movezeros(nums))


if __name__ == '__main__':
    main()