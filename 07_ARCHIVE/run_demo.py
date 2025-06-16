#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
عرض توضيحي شامل للمشروع
Comprehensive Project Demo

تشغيل جميع ميزات المشروع المطور لأفكار الباحث العلمي باسل يحيى عبدالله
Running all features of the project developed for researcher Basel Yahya Abdullah's ideas

الباحث العلمي: باسل يحيى عبدالله (Basel Yahya Abdullah)
المطور: مبتكر (Mubtakir)
التاريخ: 2025
"""

import time
import sys
from hpp_predictor import HybridPrimePredictor
from advanced_prime_algorithm import AdvancedPrimeFinder
from mathematical_analysis import MathematicalPrimeAnalysis


def print_header(title):
    """طباعة عنوان مع تنسيق جميل"""
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


def print_section(title):
    """طباعة عنوان قسم"""
    print(f"\n--- {title} ---")


def demo_dynamic_prediction():
    """عرض التنبؤ الديناميكي"""
    print_header("التنبؤ الديناميكي للأعداد الأولية")
    print("Dynamic Prime Prediction Demo")
    
    predictor = HybridPrimePredictor()
    
    # تحميل النماذج
    try:
        predictor.load_models()
        print("✓ تم تحميل النماذج بنجاح")
    except:
        print("⚠ تحذير: لم يتم العثور على النماذج المدربة")
    
    # اختبار التنبؤ الديناميكي
    test_primes = [97, 113, 127, 139, 149]
    
    print("\nاختبار التنبؤ الديناميكي:")
    print("Prime\t\tPredicted\tActual\t\tStatus")
    print("-" * 50)
    
    for p in test_primes:
        start_time = time.time()
        predicted = predictor.predict_next_prime_dynamic(p)
        end_time = time.time()
        
        # العثور على العدد الأولي الفعلي التالي
        from sympy import isprime
        actual = None
        for candidate in range(p + 1, p + 100):
            if isprime(candidate):
                actual = candidate
                break
        
        status = "✓ صحيح" if predicted == actual else "✗ خطأ"
        print(f"{p}\t\t{predicted}\t\t{actual}\t\t{status} ({end_time-start_time:.3f}s)")


def demo_advanced_sieve():
    """عرض الغربال المتقدم"""
    print_header("الغربال المقطعي المتقدم")
    print("Advanced Segmented Sieve Demo")
    
    finder = AdvancedPrimeFinder()
    
    # اختبار نطاقات مختلفة
    test_ranges = [
        (1000, 1100),
        (10000, 10200),
        (100000, 100500)
    ]
    
    print("\nاختبار الغربال المقطعي:")
    print("Range\t\t\tPrimes Found\tTime (s)")
    print("-" * 45)
    
    for start, end in test_ranges:
        start_time = time.time()
        primes = finder.advanced_segmented_sieve(start, end)
        end_time = time.time()
        
        print(f"[{start}, {end}]\t\t{len(primes)}\t\t{end_time-start_time:.4f}")
        
        # عرض بعض الأمثلة
        if len(primes) > 0:
            print(f"  أول 5: {primes[:5]}")
            if len(primes) > 5:
                print(f"  آخر 5: {primes[-5:]}")


def demo_mathematical_analysis():
    """عرض التحليل الرياضي"""
    print_header("التحليل الرياضي المتقدم")
    print("Advanced Mathematical Analysis Demo")
    
    analyzer = MathematicalPrimeAnalysis()
    
    print_section("1. تحليل نظرية الأعداد الأولية")
    x_values = [1000, 5000, 10000, 50000]
    pnt_results = analyzer.prime_number_theorem_analysis(x_values)
    
    print("x\t\tπ(x)\t\tEstimate\tError%")
    print("-" * 40)
    for i, x in enumerate(x_values):
        actual = pnt_results['actual_pi_x'][i]
        estimate = pnt_results['improved_estimate'][i]
        error = pnt_results['improved_errors'][i]
        print(f"{x}\t\t{actual}\t\t{estimate:.1f}\t\t{error:.2f}%")
    
    print_section("2. اختبار حدسية جولدباخ")
    goldbach_results = analyzer.goldbach_conjecture_test(200)
    print(f"الحالات المتحققة: {goldbach_results['verified_cases']}")
    print(f"معدل النجاح: {goldbach_results['success_rate']:.2f}%")
    print(f"أمثلة: {goldbach_results['sample_decompositions'][:3]}")
    
    print_section("3. الأعداد الأولية التوأم")
    twin_results = analyzer.twin_primes_analysis(2000)
    print(f"عدد الأزواج التوأم: {twin_results['count']}")
    print(f"الكثافة: {twin_results['density']:.4f}%")
    print(f"أول 5 أزواج: {twin_results['first_10'][:5]}")
    
    print_section("4. استكشاف فرضية ريمان")
    riemann_results = analyzer.riemann_hypothesis_exploration()
    if 'error' not in riemann_results:
        print(f"أصفار زيتا المحللة: {riemann_results['zeros_analysis']['count']}")
        print(f"متوسط الصفر: {riemann_results['zeros_analysis']['mean']:.6f}")
        print(f"متوسط الفجوة: {riemann_results['gaps_analysis']['mean_gap']:.6f}")
    else:
        print(riemann_results['error'])


def demo_hilbert_polya_matrix():
    """عرض مصفوفة هيلبرت-بوليا"""
    print_header("مصفوفة هيلبرت-بوليا التجريبية")
    print("Experimental Hilbert-Pólya Matrix Demo")
    
    predictor = HybridPrimePredictor()
    
    # بناء مصفوفات بأحجام مختلفة
    matrix_sizes = [20, 30, 40]
    
    print("\nتحليل مصفوفة هيلبرت-بوليا:")
    print("Size\t\tEigenvalues\tMax Eigenvalue\tTrace")
    print("-" * 50)
    
    for size in matrix_sizes:
        start_time = time.time()
        matrix, eigenvalues = predictor.experimental_hilbert_polya_matrix(size)
        end_time = time.time()
        
        trace = matrix.trace()
        max_eigenvalue = max(eigenvalues) if len(eigenvalues) > 0 else 0
        
        print(f"{size}x{size}\t\t{len(eigenvalues)}\t\t{max_eigenvalue:.6f}\t{trace:.6f}")
        print(f"  وقت الحساب: {end_time-start_time:.3f} ثانية")
        
        # مقارنة مع أصفار زيتا
        if len(eigenvalues) > 0:
            predictor.compare_with_zeta_zeros(eigenvalues, 3)


def demo_performance_comparison():
    """مقارنة الأداء"""
    print_header("مقارنة الأداء")
    print("Performance Comparison Demo")
    
    finder = AdvancedPrimeFinder()
    
    # مقارنة الطرق المختلفة لإيجاد الأعداد الأولية
    test_numbers = [1000, 5000, 10000]
    
    print("\nمقارنة سرعة إيجاد العدد الأولي التالي:")
    print("Number\t\tNext Prime\tTime (s)")
    print("-" * 35)
    
    for num in test_numbers:
        start_time = time.time()
        next_prime = finder.find_next_prime_optimized(num)
        end_time = time.time()
        
        print(f"{num}\t\t{next_prime}\t\t{end_time-start_time:.6f}")


def main():
    """تشغيل العرض التوضيحي الشامل"""
    print("=" * 60)
    print("  مشروع الأعداد الأولية المتقدم")
    print("  Advanced Prime Numbers Project")
    print("  ")
    print("  عرض توضيحي شامل - Comprehensive Demo")
    print("  الباحث العلمي: باسل يحيى عبدالله")
    print("  Researcher: Basel Yahya Abdullah")
    print("  المطور: مبتكر - Developer: Mubtakir")
    print("=" * 60)
    
    try:
        # 1. التنبؤ الديناميكي
        demo_dynamic_prediction()
        
        # 2. الغربال المتقدم
        demo_advanced_sieve()
        
        # 3. التحليل الرياضي
        demo_mathematical_analysis()
        
        # 4. مصفوفة هيلبرت-بوليا
        demo_hilbert_polya_matrix()
        
        # 5. مقارنة الأداء
        demo_performance_comparison()
        
        # الخلاصة
        print_header("خلاصة العرض التوضيحي")
        print("Demo Summary")
        
        print("\n✅ تم تشغيل جميع الميزات بنجاح!")
        print("✅ All features executed successfully!")
        
        print("\nالميزات المطبقة:")
        print("- التنبؤ الديناميكي بالأعداد الأولية")
        print("- الغربال المقطعي المتقدم")
        print("- التحليل الرياضي الشامل")
        print("- مصفوفة هيلبرت-بوليا التجريبية")
        print("- مقارنة الأداء والسرعة")
        
        print("\nImplemented Features:")
        print("- Dynamic prime prediction")
        print("- Advanced segmented sieve")
        print("- Comprehensive mathematical analysis")
        print("- Experimental Hilbert-Pólya matrix")
        print("- Performance comparison")
        
        print("\n" + "=" * 60)
        print("  شكراً لاستخدام المشروع!")
        print("  Thank you for using the project!")
        print("=" * 60)
        
    except KeyboardInterrupt:
        print("\n\n⚠ تم إيقاف العرض التوضيحي بواسطة المستخدم")
        print("⚠ Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ خطأ في العرض التوضيحي: {str(e)}")
        print(f"❌ Demo error: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
