#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
خوارزمية إيجاد الأعداد الأولية المتقدمة
Advanced Prime Finding Algorithm

تطبيق مباشر لأفكار الباحث العلمي باسل يحيى عبدالله الموضحة في التقرير النهائي
Direct implementation of researcher Basel Yahya Abdullah's ideas described in the final report

الباحث العلمي: باسل يحيى عبدالله (Basel Yahya Abdullah)
المطور: مبتكر (Mubtakir)
التاريخ: 2025
"""

import math
import time
from typing import List, Set, Tuple, Optional
import numpy as np
from sympy import isprime


class AdvancedPrimeFinder:
    """
    تطبيق الخوارزمية المتقدمة لإيجاد الأعداد الأولية
    كما هو موضح في التقرير النهائي
    """
    
    def __init__(self):
        self.cache = {}  # ذاكرة التخزين المؤقت للنتائج
        self.base_primes_cache = {}
        
    def sieve_of_eratosthenes(self, limit: int) -> List[int]:
        """
        غربال إراتوستينس الكلاسيكي
        Classical Sieve of Eratosthenes
        """
        if limit < 2:
            return []
            
        # إنشاء مصفوفة منطقية
        sieve = [True] * (limit + 1)
        sieve[0] = sieve[1] = False
        
        # تطبيق الغربال
        for i in range(2, int(math.sqrt(limit)) + 1):
            if sieve[i]:
                # إزالة جميع مضاعفات i
                for j in range(i * i, limit + 1, i):
                    sieve[j] = False
        
        # جمع الأعداد الأولية
        primes = [i for i in range(2, limit + 1) if sieve[i]]
        return primes
    
    def advanced_segmented_sieve(self, start: int, end: int) -> List[int]:
        """
        الغربال المقطعي المتقدم
        Advanced Segmented Sieve
        
        تطبيق الخوارزمية كما هو موضح في التقرير:
        1. إيجاد الأعداد الأولية الأساسية حتى √end
        2. تقسيم النطاق إلى مقاطع
        3. تطبيق الغربال على كل مقطع
        """
        print(f"Advanced Segmented Sieve: [{start}, {end}]")
        
        if start < 2:
            start = 2
        if end < start:
            return []
        
        # الخطوة 1: إيجاد الأعداد الأولية الأساسية
        sqrt_end = int(math.sqrt(end)) + 1
        base_primes = self.sieve_of_eratosthenes(sqrt_end)
        print(f"Base primes up to √{end} = {sqrt_end}: {len(base_primes)} primes")
        
        # الخطوة 2: إذا كان النطاق صغيراً، استخدم الغربال العادي
        if end - start <= 10000:
            return self._simple_range_sieve(start, end, base_primes)
        
        # الخطوة 3: الغربال المقطعي
        segment_size = 32768  # حجم المقطع الأمثل
        all_primes = []
        
        # إضافة الأعداد الأولية الأساسية إذا كانت في النطاق
        for p in base_primes:
            if start <= p <= end:
                all_primes.append(p)
        
        # معالجة المقاطع
        current_start = max(sqrt_end + 1, start)
        
        while current_start <= end:
            current_end = min(current_start + segment_size - 1, end)
            
            # غربلة المقطع الحالي
            segment_primes = self._sieve_segment(current_start, current_end, base_primes)
            all_primes.extend(segment_primes)
            
            print(f"Segment [{current_start}, {current_end}]: {len(segment_primes)} primes")
            current_start = current_end + 1
        
        return sorted(all_primes)
    
    def _simple_range_sieve(self, start: int, end: int, base_primes: List[int]) -> List[int]:
        """غربال بسيط للنطاقات الصغيرة"""
        sieve = [True] * (end - start + 1)
        
        for p in base_primes:
            # إيجاد أول مضاعف لـ p في النطاق
            first_multiple = max(p * p, (start + p - 1) // p * p)
            
            # غربلة جميع مضاعفات p
            for multiple in range(first_multiple, end + 1, p):
                if multiple >= start:
                    sieve[multiple - start] = False
        
        # جمع الأعداد الأولية
        primes = []
        for i in range(max(2, start), end + 1):
            if sieve[i - start]:
                primes.append(i)
        
        return primes
    
    def _sieve_segment(self, start: int, end: int, base_primes: List[int]) -> List[int]:
        """غربلة مقطع واحد"""
        sieve = [True] * (end - start + 1)
        
        for p in base_primes:
            # إيجاد أول مضاعف لـ p في هذا المقطع
            first_multiple = max(p * p, (start + p - 1) // p * p)
            
            # غربلة جميع مضاعفات p في هذا المقطع
            for multiple in range(first_multiple, end + 1, p):
                sieve[multiple - start] = False
        
        # جمع الأعداد الأولية من هذا المقطع
        segment_primes = []
        for i in range(start, end + 1):
            if sieve[i - start]:
                segment_primes.append(i)
        
        return segment_primes
    
    def find_next_prime_optimized(self, n: int) -> Optional[int]:
        """
        إيجاد العدد الأولي التالي بطريقة محسنة
        Optimized next prime finding
        
        تطبيق الخوارزمية المحسنة من التقرير
        """
        if n < 2:
            return 2
        
        # للأعداد الصغيرة، استخدم الطريقة المباشرة
        if n < 1000:
            candidate = n + 1
            while not isprime(candidate):
                candidate += 1
            return candidate
        
        # للأعداد الكبيرة، استخدم الغربال المقطعي
        search_window = max(100, int(n * 0.01))  # نافذة بحث ديناميكية
        
        start_search = n + 1
        end_search = start_search + search_window
        
        while True:
            primes_in_window = self.advanced_segmented_sieve(start_search, end_search)
            
            if primes_in_window:
                return primes_in_window[0]  # أول عدد أولي في النافذة
            
            # توسيع نافذة البحث
            start_search = end_search + 1
            search_window *= 2  # مضاعفة حجم النافذة
            end_search = start_search + search_window
            
            # حماية من الحلقات اللانهائية
            if search_window > 1000000:
                print("Warning: Search window became too large")
                break
        
        return None
    
    def prime_counting_function_approximation(self, x: float) -> float:
        """
        تقدير دالة عد الأعداد الأولية π(x)
        Prime counting function approximation
        
        استخدام التقدير المحسن: π(x) ≈ x / (ln(x) - 1.045)
        """
        if x < 2:
            return 0
        
        if x < 17:
            # قيم دقيقة للأعداد الصغيرة
            exact_values = {2: 1, 3: 2, 5: 3, 7: 4, 11: 5, 13: 6, 17: 7}
            for key in sorted(exact_values.keys()):
                if x < key:
                    return exact_values[key] - 1
                elif x == key:
                    return exact_values[key]
        
        # التقدير المحسن للأعداد الكبيرة
        ln_x = math.log(x)
        return x / (ln_x - 1.045)
    
    def estimate_nth_prime(self, n: int) -> int:
        """
        تقدير العدد الأولي رقم n
        Estimate the nth prime number
        
        استخدام التقدير: p_n ≈ n * (ln(n) + ln(ln(n)) - 1)
        """
        if n < 6:
            return [2, 3, 5, 7, 11][n-1]
        
        ln_n = math.log(n)
        ln_ln_n = math.log(ln_n)
        
        # التقدير المحسن
        estimate = n * (ln_n + ln_ln_n - 1 + (ln_ln_n - 2) / ln_n)
        
        return int(estimate)


def main():
    """اختبار الخوارزمية المتقدمة"""
    print("=== Advanced Prime Finding Algorithm ===")
    print("تطبيق الخوارزمية المتقدمة من التقرير النهائي")
    print("=" * 50)
    
    finder = AdvancedPrimeFinder()
    
    # اختبار 1: الغربال المقطعي المتقدم
    print("\n1. Testing Advanced Segmented Sieve:")
    start_time = time.time()
    primes_1000_2000 = finder.advanced_segmented_sieve(1000, 2000)
    end_time = time.time()
    
    print(f"Primes between 1000 and 2000: {len(primes_1000_2000)}")
    print(f"First 10: {primes_1000_2000[:10]}")
    print(f"Last 10: {primes_1000_2000[-10:]}")
    print(f"Time taken: {end_time - start_time:.4f} seconds")
    
    # اختبار 2: إيجاد العدد الأولي التالي
    print("\n2. Testing Next Prime Finding:")
    test_numbers = [100, 1000, 10000, 100000]
    
    for num in test_numbers:
        start_time = time.time()
        next_prime = finder.find_next_prime_optimized(num)
        end_time = time.time()
        
        print(f"Next prime after {num}: {next_prime} (Time: {end_time - start_time:.4f}s)")
    
    # اختبار 3: تقدير دالة عد الأعداد الأولية
    print("\n3. Testing Prime Counting Function:")
    test_values = [100, 1000, 10000, 100000]
    
    for x in test_values:
        estimated = finder.prime_counting_function_approximation(x)
        actual = len(finder.advanced_segmented_sieve(2, x))
        error = abs(estimated - actual) / actual * 100
        
        print(f"π({x}): Estimated = {estimated:.1f}, Actual = {actual}, Error = {error:.2f}%")
    
    print("\n" + "=" * 50)
    print("Advanced Prime Finding Algorithm - Test Complete")


if __name__ == "__main__":
    main()
