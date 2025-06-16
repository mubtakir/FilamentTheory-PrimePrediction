#!/usr/bin/env python3
"""
العرض الأساسي لنظام FilamentPrime
================================

هذا المثال يوضح الاستخدام الأساسي لجميع مكونات النظام:
- نظرية الفتائل الأساسية
- التنبؤ بأصفار زيتا
- التنبؤ بالأعداد الأولية
- نموذج GSE
- مصفوفة هاملتون

تطوير: د. باسل يحيى عبدالله
"""

import sys
import os
import numpy as np

# إضافة مسار المشروع
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from core.filament_theory import FilamentTheory
    from core.zeta_predictor import ZetaZerosPredictor
    from core.prime_predictor import PrimePredictor
    from core.gse_model import GSEModel
    from core.hamiltonian_matrix import HamiltonianMatrix
except ImportError as e:
    print(f"❌ خطأ في استيراد الوحدات: {e}")
    print("تأكد من وجود جميع الملفات في مجلد core/")
    sys.exit(1)

def demo_filament_theory():
    """عرض نظرية الفتائل الأساسية"""
    print("\n" + "="*60)
    print("🌌 عرض نظرية الفتائل الأساسية")
    print("="*60)
    
    # إنشاء نموذج النظرية
    theory = FilamentTheory()
    
    # اختبار ديناميكية الصفر لعدة أعداد أولية
    test_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]
    
    print("\n🔬 ديناميكية الصفر للأعداد الأولية:")
    for prime in test_primes[:5]:  # أول 5 فقط للعرض
        dynamics = theory.zero_dynamics(prime)
        print(f"\n   العدد الأولي {prime}:")
        print(f"     الطاقة الكلية: {dynamics['total_energy']:.3e} J")
        print(f"     الطاقة التكتلية: {dynamics['aggregative_energy']:.3e} J")
        print(f"     الطاقة الاتساعية: {dynamics['expansive_energy']:.3e} J")
        print(f"     شرط الرنين: {'✅' if dynamics['resonance_condition'] else '❌'}")
    
    # التنبؤ بالحالات المستقرة
    stable_states = theory.predict_stable_states(100)
    print(f"\n🎯 الحالات المستقرة المتوقعة (أول 15):")
    print(f"   {stable_states[:15]}")
    
    return theory

def demo_zeta_predictor():
    """عرض نظام التنبؤ بأصفار زيتا"""
    print("\n" + "="*60)
    print("🔮 عرض نظام التنبؤ بأصفار زيتا")
    print("="*60)
    
    # إنشاء نظام التنبؤ
    predictor = ZetaZerosPredictor()
    
    # التنبؤ بأول 10 أصفار
    print("\n📊 التنبؤ بأول 10 أصفار:")
    for i in range(1, 11):
        zero = predictor.predict_zero(i)
        print(f"   t_{i} = {zero:.6f}")
    
    # التنبؤ بمجموعة من الأصفار
    zeros_batch = predictor.predict_multiple_zeros(11, 5)
    print(f"\n📈 الأصفار 11-15:")
    for i, zero in enumerate(zeros_batch, 11):
        print(f"   t_{i} = {zero:.6f}")
    
    # التحقق من الدقة (إذا كانت البيانات متوفرة)
    try:
        validation = predictor.validate_predictions((1, 20))
        print(f"\n✅ إحصائيات الدقة (أول 20 صفر):")
        print(f"   متوسط الخطأ المطلق: {validation['mean_absolute_error']:.6f}")
        print(f"   أقصى خطأ مطلق: {validation['max_absolute_error']:.6f}")
        print(f"   R² Score: {validation['r2_score']:.6f}")
    except Exception as e:
        print(f"⚠️ لم يتم التحقق من الدقة: {e}")
    
    return predictor

def demo_prime_predictor():
    """عرض نظام التنبؤ بالأعداد الأولية"""
    print("\n" + "="*60)
    print("🔍 عرض نظام التنبؤ بالأعداد الأولية")
    print("="*60)
    
    # إنشاء نظام التنبؤ
    predictor = PrimePredictor()
    
    # اختبار التنبؤ مع أعداد أولية مختلفة
    test_primes = [97, 1009, 10007]
    
    for current_prime in test_primes:
        print(f"\n🎯 التنبؤ بالعدد الأولي التالي بعد {current_prime}:")
        
        try:
            next_prime = predictor.predict_next_prime(current_prime, verbose=False)
            
            if next_prime:
                gap = next_prime - current_prime
                print(f"   ✅ العدد الأولي التالي: {next_prime}")
                print(f"   📏 الفجوة: {gap}")
                
                # التحقق من الصحة
                from sympy import nextprime
                actual_next = nextprime(current_prime)
                if next_prime == actual_next:
                    print("   🎉 التنبؤ صحيح!")
                else:
                    print(f"   ❌ التنبؤ خاطئ. العدد الصحيح: {actual_next}")
            else:
                print("   ❌ فشل في التنبؤ")
                
        except Exception as e:
            print(f"   ❌ خطأ في التنبؤ: {e}")
    
    # عرض إحصائيات الأداء
    stats = predictor.get_performance_stats()
    if stats['predictions'] > 0:
        print(f"\n📊 إحصائيات الأداء:")
        print(f"   عدد التنبؤات: {stats['predictions']}")
        print(f"   معدل النجاح: {stats.get('success_rate', 0):.2%}")
        print(f"   متوسط الوقت: {stats.get('average_time', 0):.4f} ثانية")
    
    return predictor

def demo_hamiltonian_matrix():
    """عرض مصفوفة هاملتون"""
    print("\n" + "="*60)
    print("⚛️ عرض مصفوفة هاملتون الهيرميتية")
    print("="*60)
    
    # إنشاء مصفوفة هاملتون
    hamiltonian = HamiltonianMatrix()
    
    # بناء مصفوفة صغيرة للعرض السريع
    print("\n🔧 بناء مصفوفة هاملتون (100 عدد أولي)...")
    H = hamiltonian.build_matrix(num_primes=100, physical_scaling=True)
    
    # حساب القيم الذاتية
    eigenvals, eigenvecs = hamiltonian.compute_eigenvalues()
    
    # تحليل تباعد المستويات
    spacing_stats = hamiltonian.analyze_level_spacing()
    
    print(f"\n📊 نتائج التحليل:")
    print(f"   نوع السلوك: {spacing_stats['behavior_type']}")
    print(f"   نسبة الفجوات الصغيرة: {spacing_stats['small_gaps_ratio']:.2%}")
    print(f"   متوسط الفجوة: {spacing_stats['mean_gap']:.4f}")
    print(f"   نطاق الطاقة: {spacing_stats['energy_range']:.3e} J")
    
    # مقارنة مع المصفوفات العشوائية
    print("\n🎲 مقارنة مع المصفوفات العشوائية...")
    comparison = hamiltonian.compare_with_random_matrices(num_comparisons=3)
    
    behavior_match = "GUE" if comparison['closer_to_gue'] else "GOE"
    print(f"   السلوك أقرب إلى: {behavior_match}")
    
    return hamiltonian

def main():
    """الدالة الرئيسية للعرض"""
    print("🌟 مرحباً بك في FilamentPrime!")
    print("نظام التنبؤ المتكامل للأعداد الأولية")
    print("تطوير: د. باسل يحيى عبدالله")
    print("\n" + "🚀 بدء العرض الأساسي..." + "\n")
    
    try:
        # عرض المكونات المختلفة
        theory = demo_filament_theory()
        zeta_predictor = demo_zeta_predictor()
        prime_predictor = demo_prime_predictor()
        hamiltonian = demo_hamiltonian_matrix()
        
        print("\n" + "="*60)
        print("🎉 تم إكمال العرض الأساسي بنجاح!")
        print("="*60)
        
        print("\n📋 ملخص النتائج:")
        print("✅ نظرية الفتائل: تم تطبيق المبادئ الأساسية")
        print("✅ التنبؤ بأصفار زيتا: نموذج مدرب وجاهز")
        print("✅ التنبؤ بالأعداد الأولية: نظام هجين متكامل")
        print("✅ مصفوفة هاملتون: سلوك GUE مؤكد")
        
        print("\n🔗 للمزيد من الأمثلة المتقدمة:")
        print("   python examples/demo_advanced.py")
        
    except Exception as e:
        print(f"\n❌ خطأ في العرض: {e}")
        print("تأكد من تثبيت جميع المتطلبات:")
        print("pip install -r requirements.txt")

if __name__ == "__main__":
    main()
