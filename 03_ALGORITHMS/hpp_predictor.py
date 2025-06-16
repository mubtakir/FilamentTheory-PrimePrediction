#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
المتنبئ الهجين للأعداد الأولية - المحسن
Enhanced Hybrid Prime Predictor

تطبيق لأفكار الباحث العلمي باسل يحيى عبدالله
Implementation of researcher Basel Yahya Abdullah's ideas

يجمع بين التعلم الآلي والخوارزميات الرياضية للتنبؤ بالأعداد الأولية
Combines machine learning and mathematical algorithms for prime prediction

الباحث العلمي: باسل يحيى عبدالله (Basel Yahya Abdullah)
المطور: مبتكر (Mubtakir)
التاريخ: 2025
"""

import numpy as np
from sympy import primepi, isprime
import joblib
import math
import heapq
from typing import List, Tuple, Optional

class HybridPrimePredictor:
    def __init__(self):
        print("Initializing Hybrid Prime Predictor...")
        try:
            # تحميل النماذج والمعاملات المدربة مسبقاً
            self.popt_error = joblib.load('error_model_params.pkl')
            self.gse_params = joblib.load('gse_classifier_params.pkl')
            self.pi_cache = {} # ذاكرة تخزين مؤقت لدالة العد لتسريعها
            print("Models loaded successfully.")
        except FileNotFoundError:
            raise RuntimeError("Models not found. Please run 'train_models.py' first.")

    def load_models(self):
        """
        طريقة لتحميل النماذج (للتوافق مع التوقعات)
        النماذج تُحمل تلقائياً في __init__ لكن هذه الطريقة متاحة للاستدعاء الصريح
        """
        try:
            self.popt_error = joblib.load('error_model_params.pkl')
            self.gse_params = joblib.load('gse_classifier_params.pkl')
            print("Models reloaded successfully.")
        except FileNotFoundError:
            raise RuntimeError("Models not found. Please run 'train_models.py' first.")

    def _pi(self, n):
        # دالة العد مع ذاكرة مؤقتة
        if n in self.pi_cache:
            return self.pi_cache[n]
        result = int(primepi(n))  # تحويل إلى int لتجنب مشاكل sympy
        self.pi_cache[n] = result
        return result

    def _error_model(self, n, a, b, c):
        # نفس نموذج الخطأ من التدريب
        log_n = np.log(n + 1)
        log_log_n = np.log(np.log(n + 2))
        return a * (n / log_n) * log_log_n + b * (n / log_n) + c

    def _gse_classifier(self, n):
        # نفس نموذج التصنيف من التدريب
        p = self.gse_params
        term1 = np.sin(p['k1'] * np.log(n) + p['phi1'])
        term2 = np.cos(p['k2'] * np.log(n) + p['phi2'])
        logit = p['a'] * term1 + p['b'] * term2
        prob = 1 / (1 + np.exp(-logit))
        return prob > p['threshold']

    def _get_base_primes(self, limit: int) -> List[int]:
        """الحصول على الأعداد الأولية الأساسية حتى الحد المحدد"""
        if not hasattr(self, '_base_primes_cache'):
            self._base_primes_cache = {}

        if limit in self._base_primes_cache:
            return self._base_primes_cache[limit]

        # استخدام غربال بسيط لإيجاد الأعداد الأولية
        sieve = [True] * (limit + 1)
        sieve[0] = sieve[1] = False

        for i in range(2, int(limit**0.5) + 1):
            if sieve[i]:
                for j in range(i*i, limit + 1, i):
                    sieve[j] = False

        primes = [i for i in range(2, limit + 1) if sieve[i]]
        self._base_primes_cache[limit] = primes
        return primes

    def _next_odd_multiple(self, prime: int, after: int) -> int:
        """إيجاد أول مضاعف فردي للعدد الأولي بعد الرقم المحدد"""
        # M(p, n) = min {m ∈ ℕ | m > n, m = p * q, q ∈ ℕ, q is odd}
        if prime == 2:
            # للعدد 2، نبحث عن أول عدد زوجي بعد after
            return after + 1 if after % 2 == 1 else after + 2

        # للأعداد الأولية الفردية
        start = ((after // prime) + 1) * prime
        if start <= after:
            start += prime

        # تأكد أن النتيجة فردية
        if start % 2 == 0:
            start += prime

        return start

    def predict_next_prime_dynamic(self, p_current: int) -> Optional[int]:
        """
        التنبؤ بالعدد الأولي التالي باستخدام الخوارزمية الديناميكية
        المستوحاة من تحليل المصفوفات في التقرير
        """
        print(f"\n=== Dynamic Prime Prediction for p_current = {p_current} ===")

        # الخطوة 1: الحصول على الأعداد الأولية الأساسية
        sqrt_limit = int(math.sqrt(p_current * 10))  # نطاق أمان أوسع
        base_primes = self._get_base_primes(sqrt_limit)
        odd_primes = [p for p in base_primes if p > 2]  # الأعداد الأولية الفردية فقط

        print(f"Using {len(odd_primes)} base odd primes up to {sqrt_limit}")

        # الخطوة 2: إنشاء قائمة انتظار الأحداث المركبة
        # Priority Queue: (next_composite, prime_that_causes_it)
        events_queue = []

        for prime in odd_primes:
            next_composite = self._next_odd_multiple(prime, p_current)
            heapq.heappush(events_queue, (next_composite, prime))

        print(f"Initial composite events queue size: {len(events_queue)}")

        # الخطوة 3: البحث الديناميكي
        candidate = p_current + 2 if p_current % 2 == 1 else p_current + 1
        max_iterations = 1000  # حماية من الحلقات اللانهائية
        iteration = 0

        while iteration < max_iterations:
            iteration += 1

            if not events_queue:
                print("Warning: Events queue is empty!")
                break

            # الحدث المركب التالي
            next_composite, causing_prime = heapq.heappop(events_queue)

            print(f"Iteration {iteration}: candidate={candidate}, next_composite={next_composite}")

            # مقارنة المرشح مع الحدث المركب التالي
            if candidate < next_composite:
                # وجدنا فجوة! المرشح لم يمسه أي حدث مركب
                print(f"Gap found! Candidate {candidate} is potentially prime.")

                # تحقق نهائي من كون المرشح أولياً
                if isprime(candidate):
                    print(f"SUCCESS! Next prime found: {candidate}")
                    return candidate
                else:
                    print(f"False positive: {candidate} is not prime. Continuing...")
                    candidate += 2
                    # إعادة إدراج الحدث المركب
                    heapq.heappush(events_queue, (next_composite, causing_prime))
                    continue

            elif candidate == next_composite:
                # المرشح مركب، تجاهله وحدث الحالة
                print(f"Candidate {candidate} is composite (divisible by {causing_prime})")

                # تحديث حالة الآلة المسببة للحدث
                next_event = self._next_odd_multiple(causing_prime, next_composite)
                heapq.heappush(events_queue, (next_event, causing_prime))

                # الانتقال للمرشح التالي
                candidate += 2

            else:  # candidate > next_composite
                # هذا لا يجب أن يحدث في التطبيق الصحيح
                print(f"Warning: candidate {candidate} > next_composite {next_composite}")
                # تحديث الحدث وإعادة المحاولة
                next_event = self._next_odd_multiple(causing_prime, candidate)
                heapq.heappush(events_queue, (next_event, causing_prime))

        print(f"Dynamic prediction failed after {iteration} iterations.")
        return None

    def predict_next_prime(self, p_k):
        print(f"\n--- Predicting next prime after {p_k} ---")

        # ================== المرحلة 1: التقدير التقريبي ==================
        k = self._pi(p_k)
        n_next = k + 1

        # التنبؤ بـ t_{k+1}
        t_approx_simple = (2 * np.pi * float(n_next)) / np.log(float(n_next))
        error_correction = self._error_model(float(n_next), *self.popt_error)
        t_predicted = t_approx_simple + error_correction

        # ترجمة t إلى x (تقدير لموقع p_{k+1})
        p_guess = (t_predicted / (2 * np.pi)) * np.log(t_predicted)
        p_guess = int(p_guess)
        
        print(f"Phase 1: Guessed location of next prime is around {p_guess}.")
        
        # ================== المرحلة 2: البحث الموجه والترشيح ==================
        # تحديد نافذة بحث حول التقدير (توسيع النافذة للأعداد الصغيرة)
        search_window_radius = max(200, p_k // 2)  # نافذة أكبر للأعداد الصغيرة
        low_bound = max(p_k + 1, p_guess - search_window_radius)
        high_bound = max(p_k + 50, p_guess + search_window_radius)  # ضمان نافذة كافية
        
        print(f"Phase 2: Searching and filtering in window [{low_bound}, {high_bound}]...")
        
        strong_candidates = []
        for n in range(low_bound, high_bound):
            # مرشح أولي سريع: يجب أن يكون عدداً فردياً
            if n % 2 == 0:
                continue
            # مرشح GSE
            if self._gse_classifier(n):
                strong_candidates.append(n)
        
        print(f"Found {len(strong_candidates)} strong candidates.")

        # ================== المرحلة 3: التحقق الدقيق ==================
        print("Phase 3: Final verification using Miller-Rabin test.")
        
        for candidate in sorted(strong_candidates):
            if candidate <= p_k:
                continue
                
            # isprime في sympy تستخدم اختبار ميلر-رابين للأعداد الكبيرة
            if isprime(candidate):
                print(f"SUCCESS! Next prime found: {candidate}")
                return candidate
        
        print("Prediction failed. The search window might be too small or the models need refinement.")
        return None

    def advanced_segmented_sieve(self, start: int, end: int) -> List[int]:
        """
        تطبيق الغربال المقطعي المتقدم كما هو موضح في التقرير
        إيجاد جميع الأعداد الأولية في النطاق [start, end]
        """
        print(f"\n=== Advanced Segmented Sieve: [{start}, {end}] ===")

        if start < 2:
            start = 2

        # الخطوة 1: إيجاد الأعداد الأولية الأساسية
        sqrt_end = int(math.sqrt(end)) + 1
        base_primes = self._get_base_primes(sqrt_end)
        print(f"Base primes up to √{end} = {sqrt_end}: {len(base_primes)} primes")

        # الخطوة 2: إذا كان النطاق صغيراً، استخدم الغربال العادي
        if end - start < 10000:
            return self._simple_sieve_range(start, end)

        # الخطوة 3: الغربال المقطعي
        segment_size = 16384  # حجم المقطع
        all_primes = []

        # إضافة الأعداد الأولية الأساسية إذا كانت في النطاق
        for p in base_primes:
            if start <= p <= end:
                all_primes.append(p)

        # معالجة المقاطع
        current_start = max(sqrt_end + 1, start)
        if current_start % 2 == 0:
            current_start += 1  # ابدأ من عدد فردي

        while current_start <= end:
            current_end = min(current_start + segment_size - 1, end)

            # إنشاء مصفوفة المقطع (للأعداد الفردية فقط)
            segment_primes = self._sieve_segment(current_start, current_end, base_primes)
            all_primes.extend(segment_primes)

            print(f"Segment [{current_start}, {current_end}]: found {len(segment_primes)} primes")
            current_start = current_end + 1
            if current_start % 2 == 0:
                current_start += 1

        return sorted(all_primes)

    def _simple_sieve_range(self, start: int, end: int) -> List[int]:
        """غربال بسيط للنطاقات الصغيرة"""
        sieve = [True] * (end - start + 1)

        for i in range(2, int(math.sqrt(end)) + 1):
            # إيجاد أول مضاعف لـ i في النطاق
            first_multiple = max(i * i, (start + i - 1) // i * i)

            for j in range(first_multiple, end + 1, i):
                if j >= start:
                    sieve[j - start] = False

        primes = []
        for i in range(max(2, start), end + 1):
            if sieve[i - start]:
                primes.append(i)

        return primes

    def _sieve_segment(self, start: int, end: int, base_primes: List[int]) -> List[int]:
        """غربلة مقطع واحد باستخدام الأعداد الأولية الأساسية"""
        # مصفوفة للأعداد الفردية فقط في هذا المقطع
        if start % 2 == 0:
            start += 1

        odd_numbers = list(range(start, end + 1, 2))
        is_prime = [True] * len(odd_numbers)

        # تطبيق الغربال باستخدام كل عدد أولي أساسي
        for p in base_primes:
            if p == 2:
                continue  # تجاهل 2 لأننا نتعامل مع الأعداد الفردية فقط

            # إيجاد أول مضاعف فردي لـ p في هذا المقطع
            first_multiple = self._next_odd_multiple(p, start - 1)

            # غربلة كل مضاعفات p في هذا المقطع
            current = first_multiple
            while current <= end:
                if current in odd_numbers:
                    idx = odd_numbers.index(current)
                    is_prime[idx] = False
                current += 2 * p  # القفز للمضاعف الفردي التالي

        # جمع الأعداد الأولية
        segment_primes = []
        for i, prime_flag in enumerate(is_prime):
            if prime_flag:
                segment_primes.append(odd_numbers[i])

        return segment_primes

    def experimental_hilbert_polya_matrix(self, N: int = 100) -> Tuple[np.ndarray, np.ndarray]:
        """
        بناء مصفوفة هيلبرت-بوليا التجريبية كما هو مقترح في التقرير
        لاستكشاف الربط بين الأعداد الأولية وفرضية ريمان
        """
        print(f"\n=== Building Experimental Hilbert-Pólya Matrix ({N}x{N}) ===")

        def gcd(a, b):
            """حساب القاسم المشترك الأكبر"""
            while b:
                a, b = b, a % b
            return a

        # بناء مصفوفة التفاعل الأولي
        H = np.zeros((N, N))

        for n in range(1, N + 1):
            for m in range(1, N + 1):
                if gcd(n, m) > 1:
                    # استخدام الجذر التربيعي كما اقترح في التقرير
                    H[n-1, m-1] = 1 / math.sqrt(n * m)

        print("Computing eigenvalues...")
        # حساب القيم الذاتية (المصفوفة حقيقية ومتناظرة)
        eigenvalues = np.linalg.eigvalsh(H)
        eigenvalues = np.sort(eigenvalues)

        # إزالة القيم الصفرية أو القريبة من الصفر
        non_zero_eigenvalues = eigenvalues[eigenvalues > 1e-10]

        print(f"Found {len(non_zero_eigenvalues)} non-zero eigenvalues")
        print(f"Largest eigenvalues: {non_zero_eigenvalues[-5:]}")
        print(f"Smallest positive eigenvalues: {non_zero_eigenvalues[:5]}")

        return H, non_zero_eigenvalues

    def compare_with_zeta_zeros(self, eigenvalues: np.ndarray, num_zeros: int = 10) -> None:
        """
        مقارنة القيم الذاتية مع أصفار دالة زيتا المعروفة
        """
        print(f"\n=== Comparison with Riemann Zeta Zeros ===")

        # تحميل أصفار دالة زيتا
        try:
            zeta_zeros = np.loadtxt("zeta_zeros_1000.txt", encoding='utf-8-sig')[:num_zeros]
            print(f"Loaded {len(zeta_zeros)} zeta zeros")

            # مقارنة بسيطة - هذا مجرد استكشاف أولي
            print("\nFirst few zeta zeros vs eigenvalues:")
            print("Zeta Zeros\t\tEigenvalues")
            print("-" * 40)

            for i in range(min(len(zeta_zeros), len(eigenvalues), num_zeros)):
                print(f"{zeta_zeros[i]:.6f}\t\t{eigenvalues[-(i+1)]:.6f}")

            # حساب الارتباط
            min_len = min(len(zeta_zeros), len(eigenvalues))
            if min_len > 1:
                correlation = np.corrcoef(
                    zeta_zeros[:min_len],
                    eigenvalues[-min_len:]
                )[0, 1]
                print(f"\nCorrelation coefficient: {correlation:.6f}")

        except FileNotFoundError:
            print("zeta_zeros_1000.txt not found. Cannot perform comparison.")
        except Exception as e:
            print(f"Error in comparison: {e}")

    def matrix_based_prime_analysis(self, start_prime: int, matrix_size: int = 50) -> dict:
        """
        تحليل الأعداد الأولية باستخدام خصائص المصفوفة
        """
        print(f"\n=== Matrix-Based Prime Analysis ===")

        # بناء مصفوفة صغيرة للتحليل السريع
        H, eigenvalues = self.experimental_hilbert_polya_matrix(matrix_size)

        # تحليل خصائص المصفوفة
        trace = np.trace(H)
        determinant = np.linalg.det(H)
        rank = np.linalg.matrix_rank(H)

        # إحصائيات القيم الذاتية
        eigenvalue_stats = {
            'mean': np.mean(eigenvalues),
            'std': np.std(eigenvalues),
            'max': np.max(eigenvalues),
            'min': np.min(eigenvalues[eigenvalues > 1e-10])
        }

        analysis_results = {
            'matrix_size': matrix_size,
            'trace': trace,
            'determinant': determinant,
            'rank': rank,
            'eigenvalue_stats': eigenvalue_stats,
            'num_eigenvalues': len(eigenvalues)
        }

        print(f"Matrix trace: {trace:.6f}")
        print(f"Matrix rank: {rank}")
        print(f"Eigenvalue statistics: {eigenvalue_stats}")

        return analysis_results

# --- مثال على الاستخدام ---
if __name__ == '__main__':
    predictor = HybridPrimePredictor()

    print("=== Hybrid Prime Predictor - Enhanced Version ===")
    print("Based on the research report: Advanced Prime Finding Algorithm")
    print("=" * 60)

    # النماذج تُحمل تلقائياً في __init__
    print("✓ Models loaded automatically during initialization")

    # اختبار التنبؤ الديناميكي الجديد
    print("\n1. Testing Dynamic Prime Prediction:")
    test_primes = [97, 101, 103, 107, 109, 113, 127, 131]

    for p in test_primes:
        print(f"\n--- Testing with p = {p} ---")

        # التنبؤ الديناميكي الجديد
        next_prime_dynamic = predictor.predict_next_prime_dynamic(p)

        # التنبؤ التقليدي (إذا كانت النماذج متاحة)
        try:
            next_prime_traditional = predictor.predict_next_prime(p)
        except:
            next_prime_traditional = None

        # التحقق من الصحة
        actual_next = None
        for candidate in range(p + 1, p + 100):
            if isprime(candidate):
                actual_next = candidate
                break

        print(f"Actual next prime: {actual_next}")
        print(f"Dynamic prediction: {next_prime_dynamic}")
        print(f"Traditional prediction: {next_prime_traditional}")

        if next_prime_dynamic == actual_next:
            print("✓ Dynamic prediction: CORRECT")
        else:
            print("✗ Dynamic prediction: INCORRECT")

    # اختبار الغربال المقطعي المتقدم
    print("\n\n2. Testing Advanced Segmented Sieve:")
    start_range = 1000
    end_range = 1100

    primes_in_range = predictor.advanced_segmented_sieve(start_range, end_range)
    print(f"Primes in range [{start_range}, {end_range}]: {len(primes_in_range)}")
    print(f"First 10 primes: {primes_in_range[:10]}")

    # اختبار مصفوفة هيلبرت-بوليا التجريبية
    print("\n\n3. Testing Experimental Hilbert-Pólya Matrix:")
    matrix_size = 30  # حجم صغير للاختبار السريع

    H_matrix, eigenvalues = predictor.experimental_hilbert_polya_matrix(matrix_size)
    print(f"Matrix shape: {H_matrix.shape}")
    print(f"Number of eigenvalues: {len(eigenvalues)}")

    # مقارنة مع أصفار دالة زيتا
    predictor.compare_with_zeta_zeros(eigenvalues, num_zeros=5)

    # تحليل الأعداد الأولية باستخدام المصفوفة
    print("\n\n4. Matrix-Based Prime Analysis:")
    analysis = predictor.matrix_based_prime_analysis(113, matrix_size=25)

    print("\n" + "=" * 60)
    print("Enhanced Hybrid Prime Predictor - Test Complete")
    print("=" * 60)