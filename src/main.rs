use std::cell::RefCell;
use std::collections::VecDeque;
use std::collections::{HashMap, HashSet};
use std::i32;
use std::ops::Index;
use std::rc::Rc;

#[derive(Debug, PartialEq, Eq)]
pub struct TreeNode {
    val: i32,
    left: Option<Rc<RefCell<TreeNode>>>,
    right: Option<Rc<RefCell<TreeNode>>>,
}

impl TreeNode {
    #[inline]
    pub fn new(val: i32) -> Self {
        TreeNode {
            val,
            left: None,
            right: None,
        }
    }
}

pub fn main() {
    let mut tree: Vec<Vec<i32>> = vec![vec![]; 10];
    add_edge(&mut tree, 0, 1);
    add_edge(&mut tree, 0, 2);
    add_edge(&mut tree, 2, 3);
    add_edge(&mut tree, 2, 4);
    add_edge(&mut tree, 1, 6);
    add_edge(&mut tree, 1, 7);
    add_edge(&mut tree, 4, 8);
    add_edge(&mut tree, 4, 9);

    let depth = max_depth(&tree, 0);
    println!("Max depth: {}", depth);
}

pub fn count_painters(boards: &Vec<i32>, mid: &i32) -> i32 {
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
pub fn painter_problem(boards: Vec<i32>, painters: i32) -> i32 {
    let (mut low, mut high): (i32, i32) = (*boards.iter().max().unwrap(), boards.iter().sum());

    while low < high {
        let mid = (low + high) / 2;

        let painters_taken = count_painters(&boards, &mid);
        if painters_taken <= painters {
            high = mid;
        } else {
            low = mid + 1;
        }
    }
    low as i32
}

pub fn count_students(books: &Vec<i32>, mid: &i32) -> i32 {
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
pub fn book_allocation(books: Vec<i32>, m: i32) -> i32 {
    let (mut low, mut high): (i32, i32) = (*books.iter().max().unwrap(), books.iter().sum::<i32>());
    let mut result = high;
    println!("low: {}\nhigh: {}", low, high);
    while low < high {
        let mid = (low + high) / 2;
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

pub fn occurence_in_sorted_array(nums: Vec<i32>, target: i32) -> i32 {
    fn lower_bound(nums: &Vec<i32>, target: i32) -> i32 {
        let (mut l, mut r) = (0, nums.len());

        while l < r {
            let mid = (l + r) / 2;
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
            let mid = (l + r) / 2;
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

    let count = upper_bound - lower_bound;

    count
}

pub fn floor_and_ceiling(nums: Vec<i32>, target: i32) -> (i32, i32) {
    let (mut floor, mut ceiling) = (0, 0);
    let (mut l, mut r) = (0, nums.len() - 1);
    let mut x = 0;
    while l <= r {
        let mid = (l + r) / 2;
        println!("iteration: {}, l:{}, r:{}", x + 1, l, r);
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

pub fn lower_bound_binary_search(nums: Vec<i32>, target: i32) -> i32 {
    let (mut l, mut r) = (0, nums.len());

    while l < r {
        let mid = (l + r) / 2;

        if nums[mid] < target {
            l = mid + 1;
        } else {
            r = mid;
        }
    }
    l as i32
}

pub fn upper_bound_binary_search(nums: Vec<i32>, target: i32) -> i32 {
    let (mut l, mut r) = (0, nums.len());

    while l < r {
        let mid = (l + r) / 2;

        if nums[mid] > target {
            r = mid;
        } else {
            l = mid + 1;
        }
    }
    r as i32
}

pub fn longest_balanced_substring(s: String) -> i32 {
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

pub fn eat_bananas(piles: Vec<i32>, h: i32) -> i32 {
    let mut min = 1;
    let mut max = *piles.iter().max().unwrap();
    let mut min_energy = (max * max) * h;

    while min < max {
        let mid = (min + max) / 2;
        let mut total_hrs = 0;

        for pile in &piles {
            total_hrs += (pile + mid - 1) / mid;
        }

        if total_hrs <= h {
            let energy = (mid * mid) * total_hrs;
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

pub fn distinct_characters(s: String) -> i32 {
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

// pub fn subarry_with_distinct_sum(nums: Vec<i32>, k: i32) -> i32 {
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

pub fn two_array_modulus(nums1: Vec<i32>, nums2: Vec<i32>, t: i32) -> Vec<(i32, i32)> {
    let mut p1 = 0;
    let mut p2 = 0;
    let mut pairs: Vec<(i32, i32)> = Vec::new();

    while p1 < nums1.len() && p2 < nums2.len() {
        let diff = nums1[p1] - nums2[p2];

        if diff.abs() <= t {
            pairs.push((nums1[p1], nums2[p2]));
            p2 += 1;
        } else if nums1[p1] < nums2[p2] {
            p1 += 1;
        } else {
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

pub fn character_frequency(s: String) -> (char, i32) {
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

    return (max_char, max_freq);
}

// Problem: Given two strings s and t, return true if t is an anagram of s, and false otherwise.

// 🔍 Hints:
// Count the frequency of each character in s.

// Decrease the frequency using characters from t.

// If at the end all frequencies are zero and no extra characters, it's an anagram.

pub fn check_anagram(s: String, t: String) -> bool {
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

    if freq1 == freq2 { true } else { false }
}

// Given an integer array nums and an integer k,
// return the total number of continuous subarrays whose sum equals to k.

pub fn subarray_sum(nums: Vec<i32>, k: i32) -> i32 {
    let mut prefix_sum = HashMap::new();
    let mut sum: i32 = 0;
    let mut count = 0;

    prefix_sum.insert(0, 1);
    for i in 0..nums.len() {
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
pub fn longest_subarray_k(nums: Vec<i32>, k: i32) -> i32 {
    let mut left: usize = 0;
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
pub fn longet_subarray_sum_zero(nums: Vec<i32>) -> i32 {
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
pub fn xor_subarray(nums: Vec<i32>, target: i32) -> i32 {
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

pub fn find_reapeated_missing(nums: Vec<i32>) -> (i32, i32) {
    let mut results = (0, 0);
    let mut array_sum = 0;
    let n = nums.len() as i32;
    for i in 0..nums.len() {
        array_sum += nums[i];
        for j in (i + 1)..nums.len() {
            if nums[j] == nums[i] {
                results.0 = nums[j];
            }
        }
    }
    let range_sum = results.0 + (n * (n + 1)) / 2;
    results.1 = range_sum - array_sum;

    results
}

pub fn contains_duplicate(nums: Vec<i32>) -> bool {
    let mut frequency = HashSet::new();

    for num in nums {
        if let true = frequency.contains(&num) {
            return true;
        }

        frequency.insert(num);
    }

    return false;
}

pub fn contains_nearby_duplicate(nums: Vec<i32>, k: i32) -> bool {
    let n = nums.len();
    if k >= 0 {
        return false;
    }
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
    if t.len() != s.len() {
        return false;
    }

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
    let mut result = vec![1; nums.len()];
    let mut post = 1;
    let mut pre = 1;

    for i in 0..nums.len() {
        result[i] = pre;
        pre *= nums[i];
    }

    for i in (0..nums.len()).rev() {
        result[i] *= post;
        post *= nums[i];
    }
    result
}

pub fn dfs_algorithm(
    node: usize,
    parent: usize,
    tree: &Vec<Vec<i32>>,
    values: &Vec<i32>,
    ans: &mut Vec<i32>,
) -> i32 {
    let mut subtotal = values[node];

    for &child in &tree[node] {
        if child as usize == parent {
            continue;
        }

        subtotal += dfs_algorithm(child as usize, node, tree, values, ans);
    }
    ans[node] = subtotal;

    subtotal
}

pub fn max_tree_depth(root: Option<Rc<RefCell<TreeNode>>>) -> i32 {
    fn dfs(node: &Option<Rc<RefCell<TreeNode>>>) -> i32 {
        match node {
            None => 0,
            Some(ref_cell) => {
                let n = ref_cell.borrow();
                let left_depth = dfs(&n.left);
                let right_depth = dfs(&n.right);
                1 + left_depth.max(right_depth)
            }
        }
    }

    dfs(&root)
}

pub fn min_tree_depth(root: Option<Rc<RefCell<TreeNode>>>) -> i32 {
    fn dfs(node: &Option<Rc<RefCell<TreeNode>>>) -> i32 {
        match node {
            None => 0,
            Some(ref_cell) => {
                let n = ref_cell.borrow();
                let left_depth = dfs(&n.left);
                let right_depth = dfs(&n.right);
                if left_depth == 0 || right_depth == 0 {
                    1 + left_depth + right_depth
                } else {
                    1 + left_depth.min(right_depth)
                }
            }
        }
    }

    dfs(&root)
}

pub fn minimum_operations(nums: Vec<i32>) -> i32 {
    nums.iter()
        .filter(|&x| *x % 3 != 0)
        .collect::<Vec<&i32>>()
        .len() as i32
}

pub fn min_operations(nums: Vec<i32>, k: i32) -> i32 {
    nums.iter().sum::<i32>() % k
}

fn add_edge(tree: &mut Vec<Vec<i32>>, u: i32, v: i32) {
    tree[u as usize].push(v);
    tree[v as usize].push(u);
}

pub fn max_depth_adj_list(tree: &Vec<Vec<i32>>, parent: i32, node: i32) -> i32 {
    let mut best_child_depth = 0;

    for &child in &tree[node as usize] {
        if child == parent {
            continue;
        }
        let depth_child = max_depth_adj_list(tree, node, child);
        best_child_depth = best_child_depth.max(depth_child);
    }

    1 + best_child_depth
}

pub fn max_depth(tree: &Vec<Vec<i32>>, root: i32) -> i32 {
    max_depth_adj_list(tree, -1, root)
}

pub fn is_same_tree(p: Option<Rc<RefCell<TreeNode>>>, q: Option<Rc<RefCell<TreeNode>>>) -> bool {
    fn traverse(p: &Option<Rc<RefCell<TreeNode>>>, q: &Option<Rc<RefCell<TreeNode>>>) -> bool {
        match (p, q) {
            (Some(node1), Some(node2)) => {
                let node1 = node1.borrow();
                let node2 = node2.borrow();

                if node1.val != node2.val {
                    return false;
                } else {
                    let check_left = traverse(&node1.left, &node2.left);
                    let check_right = traverse(&node1.right, &node2.right);

                    match (check_left, check_right) {
                        (true, true) => true,
                        _ => false,
                    }
                }
            }
            (None, None) => false,
            _ => false,
        }
    }

    traverse(&p, &q)
}

// pub fn max_run_time(n: i32, batteries: Vec<i32>) -> i32 {
//     let mut batteries = batteries.clone();
//     if n > batteries.len() as i32 { return 0; }
//     if n == 1 { return batteries.iter().sum::<i32>();}

// } Do later...

pub fn invert_tree(root: Option<Rc<RefCell<TreeNode>>>) -> Option<Rc<RefCell<TreeNode>>> {
    fn invert(node: Option<Rc<RefCell<TreeNode>>>) -> Option<Rc<RefCell<TreeNode>>> {
        if let Some(ref_node) = node {
            let mut n = ref_node.borrow_mut();

            let left = n.left.take();
            let right = n.right.take();

            n.left = invert(right);
            n.right = invert(left);
            drop(n);
            Some(ref_node)
        } else {
            None
        }
    }
    invert(root)
}

pub fn is_balanced(root: Option<Rc<RefCell<TreeNode>>>) -> bool {
    fn get_height(node: &Option<Rc<RefCell<TreeNode>>>) -> i32 {
        match node {
            None => 0,
            Some(ref_cell) => {
                let n = ref_cell.borrow();

                let left_height = get_height(&n.left);
                let right_height = get_height(&n.right);

                1 + left_height.max(right_height)
            }
        }
    }

    fn check_balance(node: &Option<Rc<RefCell<TreeNode>>>) -> bool {
        match node {
            None => true,
            Some(ref_cell) => {
                let n = ref_cell.borrow();

                let left_height = get_height(&n.left);
                let right_height = get_height(&n.right);

                if (left_height - right_height).abs() > 1 {
                    return false;
                }

                return check_balance(&n.left) && check_balance(&n.right);
            }
        }
    }

    if check_balance(&root) { true } else { false }
}

pub fn word_break(s: String, word_dict: Vec<String>) -> bool {
    let n = s.len();
    let mut dict = HashSet::new();
    let mut dp = vec![false; n + 1];
    word_dict.iter().for_each(|w| {
        dict.insert(w);
    });
    dp[0] = true;

    for i in 1..=n {
        for j in 0..i {
            if dp[j] {
                let slice = &s[j..i];
                if dict.contains(&slice.to_string()) {
                    dp[i] = true;
                    break;
                }
            }
        }
    }

    dp[n]
}

pub fn longest_common_prefix(strs: Vec<String>) -> String {
    let mut prefix = strs[0].clone();

    for word in &strs {
        let word: Vec<char> = word.chars().collect();
        let mut test = String::new();
        let prefix_chars: Vec<char> = prefix.chars().collect();

        for i in 0..word.len() {
            if i >= word.len() || i >= prefix_chars.len() {
                break;
            }
            if prefix_chars[i] == word[i] {
                test.push(prefix_chars[i]);
            } else {
                break;
            }
        }
        prefix = String::from(test);
    }

    prefix.to_string()
}

pub fn level_order(root: Option<Rc<RefCell<TreeNode>>>) -> Vec<Vec<i32>> {
    let mut results: Vec<Vec<i32>> = Vec::new();

    fn order(node: &Option<Rc<RefCell<TreeNode>>>, level: usize, results: &mut Vec<Vec<i32>>) {
        if results.len() == level {
            results.push(Vec::new());
        }

        match node {
            None => {}
            Some(ref_cell) => {
                let n = ref_cell.borrow();

                results[level].push(n.val);

                order(&n.left, level + 1, results);
                order(&n.right, level + 1, results);
            }
        }
    }

    order(&root, 0, &mut results);

    results
}

pub fn bfs(root: Option<Rc<RefCell<TreeNode>>>) -> Vec<Vec<i32>> {
    let mut res = Vec::new();
    if root.is_none() {
        return res;
    }

    let mut q = VecDeque::new();
    q.push_back(root.unwrap());

    while !q.is_empty() {
        let level_size = q.len();
        let mut level_vals = Vec::with_capacity(level_size);

        for _ in 0..level_size {
            let node_rc = q.pop_front().unwrap();
            let node = node_rc.borrow();

            level_vals.push(node.val);

            if let Some(left) = &node.left {
                q.push_back(left.clone());
            }
            if let Some(right) = &node.right {
                q.push_back(right.clone());
            }
        }

        res.push(level_vals)
    }

    res
}

pub fn zigzag_level_order(root: Option<Rc<RefCell<TreeNode>>>) -> Vec<Vec<i32>> {
    let mut res = Vec::new();
    if root.is_none() {
        return res;
    }

    let mut q = VecDeque::new();
    q.push_back(root.unwrap());

    let mut left_to_right = true;
    while !q.is_empty() {
        let level_size = q.len();
        let mut level_vals = VecDeque::with_capacity(level_size);
        for _ in 0..level_size {
            let node_rc = q.pop_front().unwrap();
            let node = node_rc.borrow();

            if left_to_right == true {
                level_vals.push_back(node.val);
            } else {
                level_vals.push_front(node.val);
            }

            if let Some(left) = &node.left {
                q.push_back(left.clone());
            }
            if let Some(right) = &node.right {
                q.push_back(right.clone());
            }
        }

        res.push(level_vals.into_iter().collect::<Vec<i32>>());
        left_to_right = !left_to_right;
    }

    res
}

pub fn right_side_view(root: Option<Rc<RefCell<TreeNode>>>) -> Vec<i32> {
    let mut res = Vec::new();
    if root.is_none() {
        return res;
    }

    let mut q = VecDeque::new();
    q.push_back(root.unwrap());

    while !q.is_empty() {
        let level_size = q.len();
        let mut level_vals = Vec::new();

        for _ in 0..level_size {
            let node_rc = q.pop_front().unwrap();
            let node = node_rc.borrow();

            level_vals.push(node.val);

            if let Some(left) = &node.left {
                q.push_back(left.clone());
            }
            if let Some(right) = &node.right {
                q.push_back(right.clone());
            }
        }

        res.push(level_vals.pop().unwrap())
    }

    res
}

pub fn is_subtree(
    root: Option<Rc<RefCell<TreeNode>>>,
    sub_root: Option<Rc<RefCell<TreeNode>>>,
) -> bool {
    fn compare_subtree(
        node1: &Option<Rc<RefCell<TreeNode>>>,
        node2: &Option<Rc<RefCell<TreeNode>>>,
    ) -> bool {
        match (node1, node2) {
            (None, None) => true,
            (Some(n1), Some(n2)) => {
                let n1 = n1.borrow();
                let n2 = n2.borrow();

                if n1.val == n2.val {
                    if compare_subtree(&n1.left, &n2.left) && compare_subtree(&n1.right, &n2.right)
                    {
                        true
                    } else {
                        false
                    }
                } else {
                    false
                }
            }
            _ => false,
        }
    }

    fn bfs(root: &Option<Rc<RefCell<TreeNode>>>, sub_root: &Option<Rc<RefCell<TreeNode>>>) -> bool {
        let mut res = false;
        if root.is_none() && sub_root.is_none() {
            return false;
        }
        let sub_root_value = sub_root.clone().unwrap().borrow().val;

        let mut q = VecDeque::new();
        q.push_back(root.clone().unwrap());

        while !q.is_empty() {
            let level_size = q.len();

            for _ in 0..level_size {
                let node_rc = q.pop_front().unwrap();
                let node = node_rc.borrow();

                if node.val == sub_root_value {
                    if compare_subtree(&Some(node_rc.clone()), &sub_root) {
                        res = true;
                        break;
                    }
                }

                if let Some(left) = &node.left {
                    q.push_back(left.clone());
                }
                if let Some(right) = &node.right {
                    q.push_back(right.clone());
                }
            }
        }
        res
    }

    bfs(&root, &sub_root)
}

pub fn good_nodes(root: Option<Rc<RefCell<TreeNode>>>) -> i32 {
    let mut results = 0;
    let max_el = i32::MIN;
    fn count_nodes(node: &Option<Rc<RefCell<TreeNode>>>, results: &mut i32, mut max_el: i32) {
        match node {
            Some(ref_cell) => {
                let n = ref_cell.borrow();
                max_el = if n.val > max_el {
                    *results += 1;
                    n.val
                } else {
                    max_el
                };
                println!("{}", max_el);
                count_nodes(&n.left, results, max_el);
                count_nodes(&n.right, results, max_el);
            }
            None => {}
        }
    }
    count_nodes(&root, &mut results, max_el);
    println!("{}", max_el);
    results
}

pub fn diameter_of_binary_tree(root: Option<Rc<RefCell<TreeNode>>>) -> i32 {
    let mut diameter = i32::MIN;

    fn find_diameter(node: &Option<Rc<RefCell<TreeNode>>>, diameter: &mut i32) -> i32 {
        match node {
            None => 0,
            Some(cell) => {
                let n = cell.borrow();

                let left_depth = find_diameter(&n.left, diameter);
                let right_depth = find_diameter(&n.right, diameter);

                *diameter = (*diameter).max(left_depth + right_depth);

                1 + left_depth.max(right_depth)
            }
        }
    }
    find_diameter(&root, &mut diameter);
    diameter
}

pub fn max_path_sum(root: Option<Rc<RefCell<TreeNode>>>) -> i32 {
    let mut max_path_sum = i32::MIN;

    fn traverse_path(node: &Option<Rc<RefCell<TreeNode>>>, max_path_sum: &mut i32) -> i32 {
        match node {
            None => 0,
            Some(cell) => {
                let n = cell.borrow();

                let val = n.val.clone();
                let left_val = traverse_path(&n.left, max_path_sum).max(0);
                let right_val = traverse_path(&n.right, max_path_sum).max(0);

                let price = val + left_val + right_val;
                *max_path_sum = (*max_path_sum).max(price);

                val + left_val.max(right_val)
            }
        }
    }
    traverse_path(&root, &mut max_path_sum);
    max_path_sum
}

pub struct Codec;

impl Codec {
    pub fn new() -> Self {
        Codec
    }

    pub fn serialize(&self, root: Option<Rc<RefCell<TreeNode>>>) -> String {
        let mut data = Vec::new();

        fn dfs_serialize(node: &Option<Rc<RefCell<TreeNode>>>, data: &mut Vec<String>) {
            match node {
                None => {
                    data.push("#".to_string());
                }
                Some(cell) => {
                    let n = cell.borrow();
                    let val = n.val.clone();
                    data.push(val.to_string());
                    dfs_serialize(&n.left, data);
                    dfs_serialize(&n.right, data);
                }
            }
        }
        dfs_serialize(&root, &mut data);
        data.join(",")
    }

    pub fn deserialize(&self, data: String) -> Option<Rc<RefCell<TreeNode>>> {
        let data = data.split(",").collect();
        let mut idx = 0;
        fn dfs_deserialize(idx: &mut usize, data: &Vec<&str>) -> Option<Rc<RefCell<TreeNode>>> {
            let token = data[*idx];
            *idx += 1;

            match token {
                "#" => None,
                _ => {
                    let mut node = TreeNode::new(token.parse::<i32>().unwrap());

                    node.left = dfs_deserialize(idx, data);
                    node.right = dfs_deserialize(idx, data);

                    Some(Rc::new(RefCell::new(node)))
                }
            }
        }

        dfs_deserialize(&mut idx, &data)
    }
}

pub fn collect_leaves(root: Option<Rc<RefCell<TreeNode>>>) -> Vec<Vec<i32>> {
    let mut results = vec![];
    fn collect(node: &Option<Rc<RefCell<TreeNode>>>, results: &mut Vec<Vec<i32>>) -> i32 {
        match node {
            None => -1,
            Some(cell) => {
                let n = cell.borrow();
                let left_height = collect(&n.left, results);
                let righ_height = collect(&n.right, results);
                let height = left_height.max(righ_height) as usize;

                if results.len() == height {
                    results.push(vec![]);
                }
                results.get_mut(height).unwrap().push(n.val);
                1 + left_height.max(righ_height)
            }
        }
    }
    collect(&root, &mut results);
    results
}

pub fn build_tree(preorder: Vec<i32>, inorder: Vec<i32>) -> Option<Rc<RefCell<TreeNode>>> {
    let mut inorder_indices = HashMap::new();

    for (i, &val) in inorder.iter().enumerate() {
        inorder_indices.insert(val, i);
    }

    let mut idx: usize = 0;

    fn construct(
        preorder: &Vec<i32>,
        inorder_indices: &HashMap<i32, usize>,
        idx: &mut usize,
        in_left: usize,
        in_right: usize,
    ) -> Option<Rc<RefCell<TreeNode>>> {
        if in_left >= in_right {
            return None;
        }

        let root_val = preorder[*idx];
        *idx += 1;

        let mut node = TreeNode::new(root_val);
        let root_index = inorder_indices[&root_val];

        node.left = construct(preorder, inorder_indices, idx, in_left, root_index);
        node.right = construct(preorder, inorder_indices, idx, root_index + 1, in_right);

        Some(Rc::new(RefCell::new(node)))
    }

    construct(&preorder, &inorder_indices, &mut idx, 0, preorder.len())
}

pub fn is_valid_bst(root: Option<Rc<RefCell<TreeNode>>>) -> bool {
    fn validate(node: &Option<Rc<RefCell<TreeNode>>>, min: Option<i32>, max: Option<i32>) -> bool {
        match node {
            None => true,
            Some(cell) => {
                let n = cell.borrow();
                let val = n.val;
                if min.map_or(false, |m| val <= m) {
                    return false;
                }
                if max.map_or(false, |m| val >= m) {
                    return false;
                }

                return validate(&n.left, min, Some(val)) && validate(&n.right, Some(val), max);
            }
        }
    }

    validate(&root, None, None)
}

pub fn inorder_traversal(root: Option<Rc<RefCell<TreeNode>>>) -> Vec<i32> {
    let mut order: Vec<i32> = vec![];

    fn traverse(node: &Option<Rc<RefCell<TreeNode>>>, order: &mut Vec<i32>) {
        match node {
            None => {}
            Some(cell) => {
                let n = cell.borrow();
                traverse(&n.left, order);
                order.push(n.val);
                traverse(&n.right, order);
            }
        }
    }
    traverse(&root, &mut order);
    order
}

pub fn find_second_minimum_value(root: Option<Rc<RefCell<TreeNode>>>) -> i32 {
    let mut order = Vec::new();
    let mut min = i32::MIN;
    fn traverse(node: &Option<Rc<RefCell<TreeNode>>>, order: &mut Vec<i32>, min: &mut i32) {
        match node {
            None => {}
            Some(cell) => {
                let n = cell.borrow();
                traverse(&n.right, order, min);
                if n.val > *min && order.len() <= 2 {
                    order.push(n.val);
                    println!("{}", n.val);
                    *min = (*min).min(n.val);
                }
                traverse(&n.right, order, min);
            }
        }
    }
    traverse(&root, &mut order, &mut min);
    if order.len() < 2 {
        return -1;
    } else {
        return *order.iter().max().unwrap();
    }
}

pub fn lowest_common_ancestor(root: Option<Rc<RefCell<TreeNode>>>, p: Option<Rc<RefCell<TreeNode>>>, q: Option<Rc<RefCell<TreeNode>>>) -> Option<Rc<RefCell<TreeNode>>> {
    let p_val = p.unwrap().borrow().val;
    let q_val = q.unwrap().borrow().val;
    let mut root = root.clone();
    while let Some(cell) = root.clone() {
        let val = cell.borrow().val;

        if p_val < val && q_val < val {
            root = cell.borrow().left.clone();
        } else if p_val > val && q_val > val {
            root = cell.borrow().right.clone();
        } else {
            return Some(cell);
        }
    }
    None
}
