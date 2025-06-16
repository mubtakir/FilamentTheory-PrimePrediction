#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
التحليل الرياضي المتقدم للأعداد الأولية
Advanced Mathematical Analysis for Prime Numbers

تطبيق لأفكار الباحث العلمي باسل يحيى عبدالله في التحليل الرياضي الموضح في التقرير النهائي
Implementation of researcher Basel Yahya Abdullah's ideas in mathematical analysis from the final report

الباحث العلمي: باسل يحيى عبدالله (Basel Yahya Abdullah)
المطور: مبتكر (Mubtakir)
التاريخ: 2025
"""

import numpy as np
import matplotlib.pyplot as plt
import math
from typing import List, Tuple, Dict, Optional
from sympy import isprime, primepi
import scipy.stats as stats
from scipy.optimize import curve_fit


class MathematicalPrimeAnalysis:
    """
    تحليل رياضي متقدم للأعداد الأولية
    Advanced mathematical analysis of prime numbers
    """
    
    def __init__(self):
        self.zeta_zeros = None
        self.load_zeta_zeros()
    
    def load_zeta_zeros(self):
        """تحميل أصفار دالة زيتا"""
        try:
            self.zeta_zeros = np.loadtxt("zeta_zeros_1000.txt", encoding='utf-8-sig')
            print(f"Loaded {len(self.zeta_zeros)} Riemann zeta zeros")
        except FileNotFoundError:
            print("Warning: zeta_zeros_1000.txt not found")
            self.zeta_zeros = None
    
    def prime_gap_analysis(self, primes: List[int]) -> Dict:
        """
        تحليل الفجوات بين الأعداد الأولية
        Prime gap analysis
        """
        if len(primes) < 2:
            return {}
        
        gaps = [primes[i+1] - primes[i] for i in range(len(primes)-1)]
        
        analysis = {
            'gaps': gaps,
            'mean_gap': np.mean(gaps),
            'std_gap': np.std(gaps),
            'min_gap': min(gaps),
            'max_gap': max(gaps),
            'median_gap': np.median(gaps),
            'gap_distribution': {}
        }
        
        # توزيع الفجوات
        unique_gaps, counts = np.unique(gaps, return_counts=True)
        for gap, count in zip(unique_gaps, counts):
            analysis['gap_distribution'][gap] = count
        
        return analysis
    
    def bertrand_postulate_verification(self, n_max: int = 1000) -> Dict:
        """
        التحقق من مسلمة برتراند
        Bertrand's postulate verification
        
        مسلمة برتراند: لكل عدد طبيعي n > 1، يوجد عدد أولي p بحيث n < p < 2n
        """
        print(f"Verifying Bertrand's postulate up to n = {n_max}")
        
        violations = []
        verified_cases = 0
        
        for n in range(2, n_max + 1):
            # البحث عن عدد أولي في النطاق (n, 2n)
            found_prime = False
            for candidate in range(n + 1, 2 * n):
                if isprime(candidate):
                    found_prime = True
                    break
            
            if found_prime:
                verified_cases += 1
            else:
                violations.append(n)
        
        return {
            'n_max': n_max,
            'verified_cases': verified_cases,
            'violations': violations,
            'success_rate': verified_cases / (n_max - 1) * 100
        }
    
    def prime_number_theorem_analysis(self, x_values: List[int]) -> Dict:
        """
        تحليل نظرية الأعداد الأولية
        Prime Number Theorem analysis
        
        π(x) ~ x / ln(x)
        """
        results = {
            'x_values': x_values,
            'actual_pi_x': [],
            'pnt_estimate': [],
            'improved_estimate': [],
            'errors': [],
            'improved_errors': []
        }
        
        for x in x_values:
            # القيمة الفعلية لـ π(x)
            actual = int(primepi(x))
            
            # تقدير نظرية الأعداد الأولية الكلاسيكي
            pnt_est = x / math.log(x) if x > 1 else 0
            
            # التقدير المحسن
            improved_est = self._improved_prime_counting_estimate(x)
            
            # حساب الأخطاء
            error = abs(actual - pnt_est) / actual * 100 if actual > 0 else 0
            improved_error = abs(actual - improved_est) / actual * 100 if actual > 0 else 0
            
            results['actual_pi_x'].append(actual)
            results['pnt_estimate'].append(pnt_est)
            results['improved_estimate'].append(improved_est)
            results['errors'].append(error)
            results['improved_errors'].append(improved_error)
        
        return results
    
    def _improved_prime_counting_estimate(self, x: float) -> float:
        """تقدير محسن لدالة عد الأعداد الأولية"""
        if x < 2:
            return 0
        
        ln_x = math.log(x)
        
        # التقدير المحسن: π(x) ≈ x / (ln(x) - 1.045)
        return x / (ln_x - 1.045)
    
    def goldbach_conjecture_test(self, n_max: int = 1000) -> Dict:
        """
        اختبار حدسية جولدباخ
        Goldbach conjecture test
        
        كل عدد زوجي أكبر من 2 يمكن كتابته كمجموع عددين أوليين
        """
        print(f"Testing Goldbach conjecture up to {n_max}")
        
        verified_cases = []
        violations = []
        
        for n in range(4, n_max + 1, 2):  # الأعداد الزوجية فقط
            found_decomposition = False
            
            for p in range(2, n // 2 + 1):
                if isprime(p) and isprime(n - p):
                    verified_cases.append((n, p, n - p))
                    found_decomposition = True
                    break
            
            if not found_decomposition:
                violations.append(n)
        
        return {
            'n_max': n_max,
            'verified_cases': len(verified_cases),
            'violations': violations,
            'success_rate': len(verified_cases) / ((n_max - 2) // 2) * 100,
            'sample_decompositions': verified_cases[:10]  # أول 10 أمثلة
        }
    
    def twin_primes_analysis(self, limit: int = 10000) -> Dict:
        """
        تحليل الأعداد الأولية التوأم
        Twin primes analysis
        
        الأعداد الأولية التوأم: أزواج من الأعداد الأولية تختلف بـ 2
        """
        print(f"Analyzing twin primes up to {limit}")
        
        twin_primes = []
        
        for p in range(3, limit - 2, 2):  # البدء من 3 والتحرك بخطوات من 2
            if isprime(p) and isprime(p + 2):
                twin_primes.append((p, p + 2))
        
        # تحليل التوزيع
        gaps_between_twins = []
        if len(twin_primes) > 1:
            for i in range(len(twin_primes) - 1):
                gap = twin_primes[i+1][0] - twin_primes[i][0]
                gaps_between_twins.append(gap)
        
        return {
            'limit': limit,
            'twin_primes': twin_primes,
            'count': len(twin_primes),
            'density': len(twin_primes) / (limit / 2) * 100,  # كثافة تقريبية
            'gaps_between_twins': gaps_between_twins,
            'mean_gap': np.mean(gaps_between_twins) if gaps_between_twins else 0,
            'first_10': twin_primes[:10],
            'last_10': twin_primes[-10:] if len(twin_primes) >= 10 else twin_primes
        }
    
    def riemann_hypothesis_exploration(self) -> Dict:
        """
        استكشاف فرضية ريمان
        Riemann Hypothesis exploration
        """
        if self.zeta_zeros is None:
            return {'error': 'Zeta zeros not available'}
        
        print("Exploring Riemann Hypothesis connections")
        
        # تحليل أصفار دالة زيتا
        zeros_analysis = {
            'count': len(self.zeta_zeros),
            'mean': np.mean(self.zeta_zeros),
            'std': np.std(self.zeta_zeros),
            'min': np.min(self.zeta_zeros),
            'max': np.max(self.zeta_zeros)
        }
        
        # تحليل الفجوات بين الأصفار
        zero_gaps = [self.zeta_zeros[i+1] - self.zeta_zeros[i] 
                     for i in range(len(self.zeta_zeros)-1)]
        
        gaps_analysis = {
            'mean_gap': np.mean(zero_gaps),
            'std_gap': np.std(zero_gaps),
            'min_gap': np.min(zero_gaps),
            'max_gap': np.max(zero_gaps)
        }
        
        return {
            'zeros_analysis': zeros_analysis,
            'gaps_analysis': gaps_analysis,
            'first_10_zeros': self.zeta_zeros[:10].tolist(),
            'first_10_gaps': zero_gaps[:10]
        }
    
    def prime_distribution_visualization(self, limit: int = 1000):
        """
        تصور توزيع الأعداد الأولية
        Prime distribution visualization
        """
        # إيجاد الأعداد الأولية
        primes = [p for p in range(2, limit + 1) if isprime(p)]
        
        # إنشاء الرسوم البيانية
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # الرسم 1: توزيع الأعداد الأولية
        ax1.scatter(primes, [1] * len(primes), alpha=0.6, s=10)
        ax1.set_xlabel('Number')
        ax1.set_ylabel('Prime')
        ax1.set_title(f'Prime Distribution up to {limit}')
        ax1.grid(True, alpha=0.3)
        
        # الرسم 2: دالة عد الأعداد الأولية
        x_vals = list(range(2, limit + 1, 10))
        pi_x_vals = [len([p for p in primes if p <= x]) for x in x_vals]
        pnt_vals = [x / math.log(x) for x in x_vals]
        
        ax2.plot(x_vals, pi_x_vals, 'b-', label='π(x) actual', linewidth=2)
        ax2.plot(x_vals, pnt_vals, 'r--', label='x/ln(x) estimate', linewidth=2)
        ax2.set_xlabel('x')
        ax2.set_ylabel('π(x)')
        ax2.set_title('Prime Counting Function')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # الرسم 3: الفجوات بين الأعداد الأولية
        gaps = [primes[i+1] - primes[i] for i in range(len(primes)-1)]
        ax3.hist(gaps, bins=20, alpha=0.7, edgecolor='black')
        ax3.set_xlabel('Gap Size')
        ax3.set_ylabel('Frequency')
        ax3.set_title('Prime Gap Distribution')
        ax3.grid(True, alpha=0.3)
        
        # الرسم 4: الأعداد الأولية التوأم
        twin_primes = [(primes[i], primes[i+1]) for i in range(len(primes)-1) 
                       if primes[i+1] - primes[i] == 2]
        if twin_primes:
            twin_x = [tp[0] for tp in twin_primes]
            twin_y = [1] * len(twin_primes)
            ax4.scatter(twin_x, twin_y, color='red', alpha=0.7, s=20)
            ax4.set_xlabel('First Twin Prime')
            ax4.set_ylabel('Twin Prime Pair')
            ax4.set_title(f'Twin Primes up to {limit} (Count: {len(twin_primes)})')
            ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('prime_analysis_visualization.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return {
            'total_primes': len(primes),
            'twin_primes_count': len(twin_primes) if twin_primes else 0,
            'largest_gap': max(gaps) if gaps else 0,
            'mean_gap': np.mean(gaps) if gaps else 0
        }


def main():
    """تشغيل التحليل الرياضي المتقدم"""
    print("=== Advanced Mathematical Analysis of Prime Numbers ===")
    print("التحليل الرياضي المتقدم للأعداد الأولية")
    print("=" * 60)
    
    analyzer = MathematicalPrimeAnalysis()
    
    # اختبار 1: تحليل نظرية الأعداد الأولية
    print("\n1. Prime Number Theorem Analysis:")
    x_values = [100, 500, 1000, 5000, 10000]
    pnt_results = analyzer.prime_number_theorem_analysis(x_values)
    
    print("x\t\tπ(x)\t\tx/ln(x)\t\tImproved\tError%\tImproved Error%")
    print("-" * 80)
    for i, x in enumerate(x_values):
        print(f"{x}\t\t{pnt_results['actual_pi_x'][i]}\t\t"
              f"{pnt_results['pnt_estimate'][i]:.1f}\t\t"
              f"{pnt_results['improved_estimate'][i]:.1f}\t\t"
              f"{pnt_results['errors'][i]:.2f}\t{pnt_results['improved_errors'][i]:.2f}")
    
    # اختبار 2: حدسية جولدباخ
    print("\n2. Goldbach Conjecture Test:")
    goldbach_results = analyzer.goldbach_conjecture_test(100)
    print(f"Verified cases: {goldbach_results['verified_cases']}")
    print(f"Success rate: {goldbach_results['success_rate']:.2f}%")
    print(f"Sample decompositions: {goldbach_results['sample_decompositions'][:5]}")
    
    # اختبار 3: الأعداد الأولية التوأم
    print("\n3. Twin Primes Analysis:")
    twin_results = analyzer.twin_primes_analysis(1000)
    print(f"Twin primes found: {twin_results['count']}")
    print(f"Density: {twin_results['density']:.4f}%")
    print(f"Mean gap between twin pairs: {twin_results['mean_gap']:.2f}")
    print(f"First 5 twin pairs: {twin_results['first_10'][:5]}")
    
    # اختبار 4: استكشاف فرضية ريمان
    print("\n4. Riemann Hypothesis Exploration:")
    riemann_results = analyzer.riemann_hypothesis_exploration()
    if 'error' not in riemann_results:
        print(f"Zeta zeros analyzed: {riemann_results['zeros_analysis']['count']}")
        print(f"Mean zero: {riemann_results['zeros_analysis']['mean']:.6f}")
        print(f"Mean gap between zeros: {riemann_results['gaps_analysis']['mean_gap']:.6f}")
    else:
        print(riemann_results['error'])
    
    print("\n" + "=" * 60)
    print("Mathematical Analysis Complete")


if __name__ == "__main__":
    main()
