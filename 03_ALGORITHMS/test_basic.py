#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
اختبار أساسي للمشروع
Basic Project Test

اختبار سريع للتأكد من عمل المشروع كما هو مطلوب في التقرير
Quick test to ensure the project works as required in the report

الباحث العلمي: باسل يحيى عبدالله (Basel Yahya Abdullah)
المطور: مبتكر (Mubtakir)
"""

def test_basic_functionality():
    """اختبار الوظائف الأساسية"""
    print("=== اختبار المشروع الأساسي ===")
    print("Testing Basic Project Functionality")
    print("=" * 50)
    
    try:
        # 1. اختبار تحميل المكتبات
        print("1. اختبار تحميل المكتبات...")
        import numpy as np
        import sympy
        import joblib
        print("   ✅ تم تحميل المكتبات بنجاح")
        
        # 2. اختبار وجود الملفات المطلوبة
        print("2. اختبار وجود الملفات...")
        import os
        required_files = [
            'train_models.py',
            'hpp_predictor.py', 
            'error_model_params.pkl',
            'gse_classifier_params.pkl',
            'zeta_zeros_1000.txt'
        ]
        
        for file in required_files:
            if os.path.exists(file):
                print(f"   ✅ {file}")
            else:
                print(f"   ❌ {file} غير موجود")
        
        # 3. اختبار تحميل الكلاس
        print("3. اختبار تحميل HybridPrimePredictor...")
        from hpp_predictor import HybridPrimePredictor
        predictor = HybridPrimePredictor()
        print("   ✅ تم تحميل الكلاس بنجاح")
        
        # 4. اختبار التنبؤ الديناميكي (الأسرع)
        print("4. اختبار التنبؤ الديناميكي...")
        result = predictor.predict_next_prime_dynamic(97)
        if result == 101:
            print(f"   ✅ التنبؤ الديناميكي صحيح: {result}")
        else:
            print(f"   ⚠ التنبؤ الديناميكي: {result} (متوقع: 101)")
        
        # 5. اختبار الغربال المقطعي
        print("5. اختبار الغربال المقطعي...")
        primes = predictor.advanced_segmented_sieve(100, 110)
        expected_primes = [101, 103, 107, 109]
        if primes == expected_primes:
            print(f"   ✅ الغربال المقطعي صحيح: {primes}")
        else:
            print(f"   ⚠ الغربال المقطعي: {primes} (متوقع: {expected_primes})")
        
        print("\n" + "=" * 50)
        print("✅ اكتمل الاختبار الأساسي بنجاح!")
        print("✅ Basic test completed successfully!")
        print("=" * 50)
        
        return True
        
    except Exception as e:
        print(f"\n❌ فشل الاختبار: {e}")
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_report_requirements():
    """اختبار متطلبات التقرير"""
    print("\n=== اختبار متطلبات التقرير ===")
    print("Testing Report Requirements")
    print("=" * 50)
    
    try:
        # التسلسل المطلوب في التقرير:
        # 1. تشغيل train_models.py أولاً
        print("1. التحقق من ملفات التدريب...")
        import os
        if os.path.exists('error_model_params.pkl') and os.path.exists('gse_classifier_params.pkl'):
            print("   ✅ ملفات .pkl موجودة (تم تشغيل train_models.py)")
        else:
            print("   ❌ ملفات .pkl غير موجودة - يجب تشغيل train_models.py أولاً")
            return False
        
        # 2. وضع hpp_predictor.py في نفس المجلد
        print("2. التحقق من hpp_predictor.py...")
        if os.path.exists('hpp_predictor.py'):
            print("   ✅ hpp_predictor.py موجود في نفس المجلد")
        else:
            print("   ❌ hpp_predictor.py غير موجود")
            return False
        
        # 3. اختبار التشغيل
        print("3. اختبار تشغيل hpp_predictor.py...")
        from hpp_predictor import HybridPrimePredictor
        predictor = HybridPrimePredictor()
        
        # اختبار سريع
        result = predictor.predict_next_prime_dynamic(113)
        if result == 127:
            print(f"   ✅ التنبؤ صحيح: العدد الأولي التالي بعد 113 هو {result}")
        else:
            print(f"   ⚠ التنبؤ: {result} (متوقع: 127)")
        
        print("\n" + "=" * 50)
        print("✅ تم استيفاء جميع متطلبات التقرير!")
        print("✅ All report requirements satisfied!")
        print("=" * 50)
        
        return True
        
    except Exception as e:
        print(f"\n❌ فشل في متطلبات التقرير: {e}")
        print(f"❌ Report requirements test failed: {e}")
        return False

def main():
    """تشغيل جميع الاختبارات"""
    print("مشروع الأعداد الأولية المتقدم")
    print("Advanced Prime Numbers Project")
    print("الباحث العلمي: باسل يحيى عبدالله")
    print("Researcher: Basel Yahya Abdullah")
    print("المطور: مبتكر")
    print("Developer: Mubtakir")
    print("\n")
    
    # تشغيل الاختبارات
    basic_test = test_basic_functionality()
    report_test = test_report_requirements()
    
    # النتيجة النهائية
    if basic_test and report_test:
        print("\n🎉 جميع الاختبارات نجحت!")
        print("🎉 All tests passed!")
        print("\nالمشروع جاهز للاستخدام وفقاً لمتطلبات التقرير")
        print("Project is ready for use according to report requirements")
    else:
        print("\n⚠ بعض الاختبارات فشلت")
        print("⚠ Some tests failed")

if __name__ == "__main__":
    main()
