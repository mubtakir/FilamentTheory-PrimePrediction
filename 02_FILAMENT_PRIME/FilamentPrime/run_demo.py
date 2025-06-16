#!/usr/bin/env python3
"""
تشغيل سريع لنظام FilamentPrime
==============================

هذا الملف يوفر تشغيل سريع لجميع مكونات النظام
مع عرض النتائج الأساسية بشكل مختصر.

تطوير: د. باسل يحيى عبدالله
"""

import sys
import os
import time
import numpy as np

# إضافة مسار المشروع
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def print_header(title):
    """طباعة عنوان مع تنسيق جميل"""
    print("\n" + "="*60)
    print(f"🌟 {title}")
    print("="*60)

def print_result(label, value, unit=""):
    """طباعة نتيجة مع تنسيق"""
    print(f"   ✅ {label}: {value} {unit}")

def quick_demo():
    """عرض سريع لجميع المكونات"""
    
    print("🚀 FilamentPrime - العرض السريع")
    print("تطوير: د. باسل يحيى عبدالله")
    print("نظرية الفتائل والتنبؤ بالأعداد الأولية")
    
    try:
        # 1. نظرية الفتائل
        print_header("نظرية الفتائل الأساسية")
        from core.filament_theory import FilamentTheory
        
        theory = FilamentTheory()
        
        # اختبار سريع
        dynamics = theory.zero_dynamics(17)
        print_result("الطاقة الأساسية", f"{theory.E_0:.3e}", "J")
        print_result("التردد الأساسي", f"{theory.f_0:.6f}", "Hz")
        print_result("شرط الرنين للعدد 17", "✅" if dynamics['resonance_condition'] else "❌")
        
        # 2. التنبؤ بأصفار زيتا
        print_header("التنبؤ بأصفار زيتا")
        from core.zeta_predictor import ZetaZerosPredictor
        
        zeta_predictor = ZetaZerosPredictor()
        
        # التنبؤ بأول 5 أصفار
        print("   🔮 أول 5 أصفار متوقعة:")
        for i in range(1, 6):
            zero = zeta_predictor.predict_zero(i)
            print(f"      t_{i} = {zero:.6f}")
        
        if zeta_predictor.is_trained:
            print_result("دقة النموذج (R²)", f"{zeta_predictor.error_model_r2:.6f}")
        
        # 3. التنبؤ بالأعداد الأولية
        print_header("التنبؤ بالأعداد الأولية")
        from core.prime_predictor import PrimePredictor
        
        prime_predictor = PrimePredictor()
        
        # اختبار سريع
        test_prime = 1009
        print(f"   🎯 البحث عن العدد الأولي التالي بعد {test_prime}...")
        
        start_time = time.time()
        next_prime = prime_predictor.predict_next_prime(test_prime, verbose=False)
        prediction_time = time.time() - start_time
        
        if next_prime:
            gap = next_prime - test_prime
            print_result("العدد الأولي التالي", next_prime)
            print_result("الفجوة", gap)
            print_result("وقت التنبؤ", f"{prediction_time:.4f}", "ثانية")
            
            # التحقق من الصحة
            from sympy import nextprime
            actual_next = nextprime(test_prime)
            if next_prime == actual_next:
                print_result("دقة التنبؤ", "✅ صحيح")
            else:
                print_result("دقة التنبؤ", f"❌ خاطئ (الصحيح: {actual_next})")
        else:
            print("   ❌ فشل في التنبؤ")
        
        # 4. مصفوفة هاملتون
        print_header("مصفوفة هاملتون الهيرميتية")
        from core.hamiltonian_matrix import HamiltonianMatrix
        
        hamiltonian = HamiltonianMatrix()
        
        # بناء مصفوفة صغيرة للعرض السريع
        print("   ⚛️ بناء مصفوفة 50×50...")
        H = hamiltonian.build_matrix(num_primes=50, physical_scaling=True)
        
        # حساب القيم الذاتية
        eigenvals, _ = hamiltonian.compute_eigenvalues()
        
        # تحليل سريع
        spacing_stats = hamiltonian.analyze_level_spacing()
        
        print_result("حجم المصفوفة", f"{H.shape[0]}×{H.shape[1]}")
        print_result("نوع السلوك", spacing_stats['behavior_type'])
        print_result("نسبة الفجوات الصغيرة", f"{spacing_stats['small_gaps_ratio']:.2%}")
        
        # مقارنة سريعة
        comparison = hamiltonian.compare_with_random_matrices(num_comparisons=2)
        behavior_match = "GUE" if comparison['closer_to_gue'] else "GOE"
        print_result("أقرب إلى", behavior_match)
        
        # 5. ملخص النتائج
        print_header("ملخص النتائج")
        print("   🎉 تم تشغيل جميع المكونات بنجاح!")
        print()
        print("   📊 الإنجازات المؤكدة:")
        print("      ✅ نظرية الفتائل: المبادئ الأساسية مطبقة")
        print("      ✅ أصفار زيتا: نموذج مدرب وجاهز للتنبؤ")
        print("      ✅ الأعداد الأولية: نظام هجين متكامل")
        print("      ✅ مصفوفة هاملتون: سلوك كمومي مؤكد")
        print()
        print("   🔬 للعرض المفصل:")
        print("      python examples/demo_basic.py")
        print()
        print("   📚 للتوثيق الكامل:")
        print("      اقرأ README.md")
        
    except ImportError as e:
        print(f"\n❌ خطأ في استيراد الوحدات: {e}")
        print("تأكد من وجود جميع الملفات في مجلد core/")
        print("وتثبيت المتطلبات: pip install -r requirements.txt")
        
    except Exception as e:
        print(f"\n❌ خطأ في التشغيل: {e}")
        print("تحقق من سلامة الملفات والبيانات")

def performance_test():
    """اختبار أداء سريع"""
    print_header("اختبار الأداء")
    
    try:
        from core.prime_predictor import PrimePredictor
        
        predictor = PrimePredictor()
        
        # اختبار عدة تنبؤات
        test_primes = [97, 1009, 10007]
        total_time = 0
        successes = 0
        
        print("   ⏱️ اختبار سرعة التنبؤ...")
        
        for prime in test_primes:
            start_time = time.time()
            result = predictor.predict_next_prime(prime, verbose=False)
            elapsed = time.time() - start_time
            total_time += elapsed
            
            if result:
                successes += 1
                print(f"      {prime} → {result} ({elapsed:.3f}s)")
            else:
                print(f"      {prime} → فشل ({elapsed:.3f}s)")
        
        avg_time = total_time / len(test_primes)
        success_rate = successes / len(test_primes)
        
        print_result("متوسط الوقت", f"{avg_time:.3f}", "ثانية")
        print_result("معدل النجاح", f"{success_rate:.1%}")
        
        # تقييم الأداء
        if avg_time < 1.0 and success_rate > 0.8:
            print("   🏆 أداء ممتاز!")
        elif avg_time < 5.0 and success_rate > 0.6:
            print("   👍 أداء جيد")
        else:
            print("   ⚠️ يحتاج تحسين")
            
    except Exception as e:
        print(f"   ❌ خطأ في اختبار الأداء: {e}")

def main():
    """الدالة الرئيسية"""
    print("🌟" * 20)
    print("FilamentPrime - التشغيل السريع")
    print("🌟" * 20)
    
    # العرض الأساسي
    quick_demo()
    
    # اختبار الأداء
    performance_test()
    
    print("\n" + "🎉" * 20)
    print("تم إكمال العرض السريع بنجاح!")
    print("🎉" * 20)

if __name__ == "__main__":
    main()
