#=======================================================================
#  Author: Isai Damier
#  Title: Radix Sort
#  Project: geekviewpoint
#  Package: algorithms
#
#  Statement:
#  Given a disordered list of integers, rearrange them in natural order.
#
#  Sample Input: [18,5,100,3,1,19,6,0,7,4,2]
#
#  Sample Output: [0,1,2,3,4,5,6,7,18,19,100]
#
#  Time Complexity of Solution:
#  Best Case O(kn); Average Case O(kn); Worst Case O(kn),
#  where k is the length of the longest number and n is the
#  size of the input array.
#
#  Note: if k is greater than log(n) then an nlog(n) algorithm would
#  be a better fit. In reality we can always change the radix
#  to make k less than log(n).
#
#  Approach:
#  radix sort, like counting sort and bucket sort, is an integer based
#  algorithm (i.e. the values of the input array are assumed to be
#  integers). Hence radix sort is among the fastest sorting algorithms
#  around, in theory. The particular distinction for radix sort is
#  that it creates a bucket for each cipher (i.e. digit); as such,
#  similar to bucket sort, each bucket in radix sort must be a
#  growable list that may admit different keys.
#
#  For decimal values, the number of buckets is 10, as the decimal
#  system has 10 numerals/cyphers (i.e. 0,1,2,3,4,5,6,7,8,9). Then
#  the keys are continuously sorted by significant digits.
#======================================================================= 

def radixsort(input_list):
	RADIX = 10
	maxLength = False
	tmp , placement = -1, 1
 
	while not maxLength:
		maxLength = True
		# declare and initialize buckets
		buckets = [list() for _ in range(RADIX)]
	 
		# split input_list between lists
		for  i in input_list:
			tmp = i / placement
			buckets[tmp % RADIX].append(i)
			if maxLength and tmp > 0:
				maxLength = False
	 
		# empty lists into input_list array
		a = 0
		for b in range(RADIX):
			buck = buckets[b]
			for i in buck:
				input_list[a] = i
				a += 1
		 
		# move to next digit
		placement *= RADIX