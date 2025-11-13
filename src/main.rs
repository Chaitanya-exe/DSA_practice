use core::num;
use std::collections::{HashMap, HashSet};


fn main() {
    let nums = vec! [5, 5, 5, 5];
    let k = 2;
    assert_eq!(painter_problem(nums, k), 10);
    let nums = vec![10, 20, 30, 40];
    let k = 2;
    assert_eq!(painter_problem(nums, k), 60);
    println!("Tests passed successfully.");
}

fn count_painters (boards: &Vec<i32>, mid: &i32) -> i32 {
    let mut time_taken: i32 = 0;
    let mut painters = 1;
    for board in boards {
        if board + time_taken <= *mid {
            time_taken += board;
        } else {
            painters += 1;
            time_taken = *board;
        }
        
    }
    painters
}

// Painter's problem. Given an array of boards and k painters allocate the boards to painter in such a way that all boards are painted in the minimum time.
fn painter_problem(boards: Vec<i32>, painters: i32) -> i32{
    let (mut low, mut high): (i32, i32) = (*boards.iter().max().unwrap(), boards.iter().sum());

    while low < high {
        let mid = (low + high)/2;

        let painters_taken = count_painters(&boards, &mid);
        if painters_taken <= painters {
            high = mid;
        } else {
            low = mid + 1;
        }
    }
    low as i32
}

fn count_students(books: &Vec<i32>, mid: &i32) -> i32 {
    let mut pages_alloc = 0;
    let mut students = 1;
    for book in books {
        if pages_alloc + *book > *mid { 
            students += 1;
            pages_alloc = *book;
        } else {
            pages_alloc += *book
        }
    }
    return students;
}

// Book allocation problem
fn book_allocation(books: Vec<i32>, m: i32) -> i32{
    let (mut low, mut high): (i32, i32) = (*books.iter().max().unwrap(), books.iter().sum::<i32>());
    let mut result = high;
    println!("low: {}\nhigh: {}", low, high);
    while low < high {
        let mid = (low + high)/2;
        println!("{mid}");
        let students = count_students(&books, &mid);

        if students <= m {
            result = mid;
            high = mid - 1;
        } else {
            low = mid + 1;
        }
    }

    result as i32
}

fn occurence_in_sorted_array(nums: Vec<i32>, target: i32) -> i32 {
    let mut result = 0;

    fn lower_bound(nums: &Vec<i32>, target: i32) -> i32{
        let (mut l, mut r) = (0, nums.len());

        while l < r {
            let mid = (l+r)/2;
            if nums[mid] < target {
                l = mid + 1
            } else {
                r = mid;
            }
        }

        l as i32
    }

    fn upper_bound(nums: &Vec<i32>, target: i32) -> i32 {
        let (mut l, mut r) = (0, nums.len());

        while l < r {
            let mid = (l+r)/2;
            if nums[mid] <= target {
                l = mid + 1
            } else {
                r = mid;
            }
        }

        l as i32
    }

    let upper_bound = upper_bound(&nums, target);
    let lower_bound = lower_bound(&nums, target);

    let count = upper_bound - lower_bound ;

    count
}

fn floor_and_ceiling(nums: Vec<i32>, target: i32) -> (i32, i32) {
    let (mut floor, mut ceiling) = (0,0);
    let (mut l, mut r) = (0, nums.len() - 1);
    let mut x = 0;
    while l <= r {
        let mid = (l+r)/2;
        println!("iteration: {}, l:{}, r:{}", x+1,l, r);
        if nums[mid] >= target {
            r = mid - 1;
            ceiling = nums[r];
        } else if nums[mid] < target {
            l = mid + 1;
            floor = nums[l];
        }
        x += 1;
    }    

    (floor, ceiling)
}

fn lower_bound_binary_search(nums: Vec<i32>, target: i32)-> i32 {
    let (mut l, mut r) = (0, nums.len());

    while l < r {
        let mid = (l+r)/2;

        if nums[mid] < target {
            l = mid + 1;
        } else {
            r = mid; 
        }   
    }
    l as i32
}

fn upper_bound_binary_search(nums: Vec<i32>, target: i32)-> i32 {
    let (mut l, mut r) = (0, nums.len());

    while l < r {
        let mid = (l+r)/2;

        if nums[mid] > target {
            r = mid;
        } else {
            l = mid+1; 
        }   
    }
    r as i32
}


fn longest_balanced_substring(s: String) -> i32 {
    let binary = vec!['0', '1'];
    let mut count: i32 = 0;
    let mut length = 0;
    let mut prefix_sum = HashMap::new();
    let chars: Vec<char> = s.chars().collect();

    prefix_sum.insert(0, -1);
    for i in 0..chars.len() {
       if chars[i] == '0' {
            count -= 1;
       } else {
            count += 1;
       }

       if let Some(&prev) = prefix_sum.get(&count) {
            length = length.max(i as i32 - prev);
       } else {
            prefix_sum.insert(count, i as i32);
       }
    }

    return length as i32;
}

fn eat_bananas(piles: Vec<i32>, h: i32) -> i32 {
    let mut min = 1;
    let mut max = *piles.iter().max().unwrap();
    let mut min_energy = (max*max) * h;

    while min < max {
        let mid = (min + max)/2;
        let mut total_hrs = 0;
        let mut energy = 0;

        for pile in &piles {
            total_hrs += (pile + mid - 1)/mid;
        }

        if total_hrs <= h {
            energy = (mid*mid) * total_hrs;
            min_energy = min_energy.min(energy);
            max = mid;
        } else {
            min = mid + 1;
        }
    }

    return min_energy as i32;
}



// Given a string s, return the length of the longest substring that contains at most 2 distinct characters.

// 🔹 Example:
// Input: s = "eceba"
// Output: 3
// Explanation: "ece" is the longest with at most 2 distinct characters.
// 🔧 Constraints:
// Use sliding window with a hash map (or array) to track window state.

// Must run in O(n)

fn distinct_characters(s: String) -> i32{
    let mut frequency = HashMap::new();
    let mut low: usize = 0;
    let chars: Vec<char> = s.chars().collect();
    let mut max_len = 0;

    for high in 0..s.len() {
        *frequency.entry(&chars[high]).or_insert(0) += 1;
        
        if frequency.len() > 2 {
            *frequency.get_mut(&chars[low]).unwrap() -= 1;
            if frequency[&chars[low]] == 0 {
                frequency.remove(&chars[low]); 
            }
            low += 1;
        }

        max_len = max_len.max(high - low + 1);
    }

    return max_len as i32;
}



// 📝 Problem:
// Given an integer array nums, return the total number of continuous subarrays whose sum equals to k.

// 🔹 Example:
// Input: nums = [1,1,1], k = 2
// Output: 2
// 🔧 Constraints:
// Use prefix sum and a hashmap to track prefix frequencies.
// Avoid nested loops (O(n²) TLE)

// fn subarry_with_distinct_sum(nums: Vec<i32>, k: i32) -> i32 {
//     let mut prefix_sum = HashMap::new();
//     let mut sum = nums[0];
//     let mut cont_arrays = 0
//     let mut low = 0; 
//     prefix_sum.insert(0, 1);
//     for high in 0..nums.len() {
//         sum += nums[high];
//         prefix_sum.entry(sum).or_insert(0) += 1;
        
//         if sum == k {
//             cont_arrays += 1;
//         }
    
//     }

// }



// 📝 Problem:
// You are given two sorted integer arrays nums1 and nums2.
// Find all pairs (u,v) where u from nums1, v from nums2, such that:
// |u - v| <= t
// Return all such pairs. If there are multiple matches for a u, return all.

fn two_array_modulus(nums1: Vec<i32>, nums2: Vec<i32>, t: i32) -> Vec<(i32,i32)> {
    let mut p1 = 0;
    let mut p2 = 0;
    let mut pairs: Vec<(i32, i32)> = Vec::new();

    while p1 < nums1.len() && p2 < nums2.len(){
        let diff = nums1[p1] - nums2[p2];

        if diff.abs() <= t {
            pairs.push((nums1[p1], nums2[p2]));
            p2 += 1;
        }else if nums1[p1] < nums2[p2] {
            p1 += 1;
        }
        else {
            p2 += 1;
        }
    }

    pairs
} 



// Count the number of characters in a string, and return the character(s) with the highest frequency.
// We’ll turn this into a mini-project:
// Step 1: Build the frequency map.
// Step 2: Find the max value in the map.
// Step 3: Return all chars with that frequency.

fn character_frequency(s: String) -> (char, i32) {
    let mut frequency = HashMap::new();
    let mut max_freq = 0;
    let mut max_char = 'a';
    let chars: Vec<char> = s.chars().collect();

    for ch in chars {
        *frequency.entry(ch).or_insert(0) += 1;

        if frequency[&ch] > max_freq {
            max_freq = *frequency.get(&ch).unwrap();
            max_char = ch;
        }
    }

    return (max_char, max_freq)
}

// Problem: Given two strings s and t, return true if t is an anagram of s, and false otherwise.

// 🔍 Hints:
// Count the frequency of each character in s.

// Decrease the frequency using characters from t.

// If at the end all frequencies are zero and no extra characters, it's an anagram.

fn check_anagram(s: String, t: String) -> bool {
    let mut result = false;
    let mut freq1 = HashMap::new();
    let mut freq2 = HashMap::new();

    let mut p1 = 0;
    let mut p2 = 0;

    let s_chars: Vec<char> = s.chars().collect();
    let t_chars: Vec<char> = t.chars().collect();

    while p1 < s.len() && p2 < t.len() {
        if p1 < s.len() {
            *freq1.entry(s_chars[p1]).or_insert(0) += 1;
            p1 += 1;
        }
        if p2 < t.len() {
            *freq2.entry(t_chars[p2]).or_insert(0) += 1;
            p2 += 1;
        }
    }

    if freq1 == freq2 {
        result = true
    } else {
        result = false
    }
    return result
}

// Given an integer array nums and an integer k,
// return the total number of continuous subarrays whose sum equals to k.

fn subarray_sum(nums: Vec<i32>, k: i32) -> i32 {
    let mut prefix_sum = HashMap::new();
    let mut sum:i32 = 0;
    let mut count = 0;

    prefix_sum.insert(0, 1);
    for i in 0..nums.len(){
        sum += nums[i];

        if let Some(freq) = prefix_sum.get(&(sum - k)) {
            count += freq;
        }

        *prefix_sum.entry(sum).or_insert(0) += 1;

    }

    count as i32
}

// Longest Subarray with sum <= k
// Find the longest subarrray whose sum is less than or equal to a given 'k'
fn longest_subarray_k(nums: Vec<i32>, k: i32) -> i32{
    let mut left:usize = 0;
    let (mut sum, mut max_len) = (0, 0);

    for right in 0..nums.len() {
        sum += nums[right];

        while sum > k {
            sum -= nums[left];
            left += 1;
        }

        if sum <= k {
            max_len = max_len.max(right - left + 1);
        }
    }

    return max_len as i32;
}

// Given an array containing both positive and negative integers, we have to find the length of the longest subarray with the sum of all elements equal to zero.
fn longet_subarray_sum_zero(nums: Vec<i32>) -> i32{
    let mut prefix_sum = 0;
    let mut seen = HashMap::new();
    let mut max_len = 0;

    for (i, &num) in nums.iter().enumerate() {
        prefix_sum += num;

        if prefix_sum == 0 {
            max_len = i + 1;
        }

        if let Some(&first_index) = seen.get(&prefix_sum) {
            max_len = max_len.max(i - first_index);
        } else {
            seen.insert(prefix_sum, i);
        }
    }

    max_len as i32
}

//  Given an array of integers A and an integer B. Find the total number of subarrays having bitwise XOR of all elements equal to k.
fn xor_subarray(nums: Vec<i32>, target: i32) -> i32 {
    let mut prefix_xor = 0;
    let mut seen = HashMap::new();
    let mut count = 0;

    for i in 0..nums.len() {
        prefix_xor ^= nums[i];

        if prefix_xor == target {
            count += 1;
        }

        if let Some(&freq) = seen.get(&(prefix_xor ^ target)) {
            count += freq
        }

        *seen.entry(prefix_xor).or_insert(0) += 1;
    }

    count
}

fn find_reapeated_missing(nums: Vec<i32>) -> (i32, i32) {

    let mut results = (0, 0);
    let mut array_sum = 0;
    let n = nums.len() as i32;
    for i in 0..nums.len() {
        array_sum += nums[i];
        for j in (i+1)..nums.len() {
            if nums[j] == nums[i] {
                results.0 = nums[j];
            }
        }
    }
    let range_sum = results.0 + (n*(n+1))/2 ;
    results.1 = range_sum - array_sum;

    results
}

pub fn contains_duplicate(nums: Vec<i32>) -> bool {
    let mut frequency = HashSet::new();

    for num in nums {
        if let true = frequency.contains(&num) {
            return true
        } 
        
        frequency.insert(num);
    }

    return false;
}

pub fn contains_nearby_duplicate(nums: Vec<i32>, k: i32) -> bool {
    let n = nums.len();
    if k >= 0 { return false; }
    let mut seen = HashSet::with_capacity(k.saturating_add(1) as usize);
    let k = k as usize;

    for i in 0..n {
        if seen.contains(&nums[i]) {
            return true;
        }

        seen.insert(&nums[i]);

        if i >= k {
            seen.remove(&nums[i - k]);
        }
    }
    false
}

pub fn is_anagram(s: String, t: String) -> bool {
    if t.len() != s.len() { return false; }

    let mut s_freq = HashMap::new();
    let mut t_freq = HashMap::new();

    let s_chars = s.chars().collect::<Vec<char>>();
    let t_chars = t.chars().collect::<Vec<char>>();

    let mut i = 0;

    while i < s.len() {
        *s_freq.entry(&s_chars[i]).or_insert(0) += 1;
        *t_freq.entry(&t_chars[i]).or_insert(0) += 1;
        i += 1;
    }

    if s_freq == t_freq {
        return true;
    } else {
        return false;
    }
}

pub fn product_except_self(nums: Vec<i32>) -> Vec<i32> {
    let mut prefix_product = HashMap::new();
    let mut product = 1;

    for (i, num) in nums.iter().enumerate() {
        product = product *num;
        prefix_product.insert(i, product);
    }

    return vec![];  
}