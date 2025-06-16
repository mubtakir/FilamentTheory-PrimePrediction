"""
نظام التنبؤ بالأعداد الأولية
===========================

النظام الهجين المتكامل للتنبؤ بالعدد الأولي التالي
يدمج نظرية الفتائل مع نموذج GSE ومصفوفة هاملتون

تطوير: د. باسل يحيى عبدالله
"""

import numpy as np
from sympy import primepi, isprime, nextprime
import time
from typing import Optional, Dict, List, Tuple
from .filament_theory import FilamentTheory
from .zeta_predictor import ZetaZerosPredictor

class PrimePredictor:
    """
    نظام التنبؤ الهجين للأعداد الأولية
    
    يستخدم ثلاث مراحل:
    1. التقدير الأولي باستخدام أصفار زيتا
    2. الترشيح باستخدام نموذج GSE
    3. التحقق النهائي باختبار الأولية
    """
    
    def __init__(self):
        """تهيئة النظام الهجين"""
        print("🚀 تهيئة نظام التنبؤ الهجين للأعداد الأولية...")
        
        # تهيئة المكونات الأساسية
        self.theory = FilamentTheory()
        self.zeta_predictor = ZetaZerosPredictor()
        
        # إحصائيات الأداء
        self.performance_stats = {
            'predictions': 0,
            'successes': 0,
            'total_time': 0,
            'total_tests': 0,
            'average_gap': 0
        }
        
        # ذاكرة التخزين المؤقت
        self.prediction_cache = {}
        
        print("✅ النظام جاهز للعمل!")
    
    def _estimate_prime_from_zeta(self, k: int) -> int:
        """
        تقدير موقع العدد الأولي رقم k باستخدام أصفار زيتا
        
        Args:
            k: ترتيب العدد الأولي
            
        Returns:
            التقدير الأولي لموقع العدد الأولي
        """
        # التنبؤ بصفر زيتا المقابل
        t_k = self.zeta_predictor.predict_zero(k)
        
        # تحويل صفر زيتا إلى تقدير للعدد الأولي
        # الصيغة العكسية المحسنة
        estimate = (t_k / (2 * np.pi)) * np.log(t_k)
        
        # تصحيح إضافي للدقة
        if t_k > np.e:
            correction = np.log(np.log(t_k)) / np.log(t_k)
            estimate *= (1 + correction)
        
        return int(estimate)
    
    def _adaptive_search_window(self, prime_estimate: int, current_prime: int) -> Tuple[int, int]:
        """
        حساب نافذة البحث التكيفية
        
        Args:
            prime_estimate: التقدير الأولي
            current_prime: العدد الأولي الحالي
            
        Returns:
            (بداية النافذة، نهاية النافذة)
        """
        # حساب الفجوة المتوقعة بناءً على نظرية الأعداد الأولية
        expected_gap = np.log(current_prime) ** 2
        
        # حساب عدم اليقين في التقدير
        uncertainty = max(1000, int(0.1 * expected_gap))
        
        # نافذة البحث
        window_start = max(current_prime + 1, prime_estimate - uncertainty)
        window_end = prime_estimate + uncertainty
        
        return window_start, window_end
    
    def _gse_prime_probability(self, x: int) -> float:
        """
        حساب احتمالية كون العدد أولي باستخدام نموذج GSE مبسط
        
        Args:
            x: العدد المراد فحصه
            
        Returns:
            احتمالية كونه أولي (0-1)
        """
        if x < 2:
            return 0.0
        if x == 2:
            return 1.0
        if x % 2 == 0:
            return 0.0
        
        # نموذج GSE مبسط باستخدام أصفار زيتا المعروفة
        log_x = np.log(x)
        
        # استخدام أول أصفار زيتا كترددات
        zeta_frequencies = [14.134725, 21.022040, 25.010858, 30.424876]

        # حساب المكونات الجيبية
        gse_value = 0
        for freq in zeta_frequencies:
            gse_value += np.sin(freq * log_x) + np.cos(freq * log_x)
        
        # تحويل إلى احتمالية
        probability = 1 / (1 + np.exp(-gse_value))
        
        # تعديل بناءً على خصائص العدد
        if x % 6 in [1, 5]:  # الأعداد الأولية > 3 تكون من الشكل 6k±1
            probability *= 1.2
        
        return min(1.0, probability)
    
    def predict_next_prime(self, current_prime: int, 
                          gse_threshold: float = 0.6,
                          max_candidates: int = 1000,
                          verbose: bool = True) -> Optional[int]:
        """
        التنبؤ بالعدد الأولي التالي
        
        Args:
            current_prime: العدد الأولي الحالي
            gse_threshold: عتبة مصنف GSE
            max_candidates: أقصى عدد مرشحين
            verbose: طباعة التفاصيل
            
        Returns:
            العدد الأولي التالي أو None
        """
        if verbose:
            print(f"\n🔍 البحث عن العدد الأولي التالي بعد {current_prime:,}")
            print("-" * 60)
        
        start_time = time.time()
        
        # التحقق من الذاكرة المؤقتة
        if current_prime in self.prediction_cache:
            if verbose:
                print("💾 تم العثور على النتيجة في الذاكرة المؤقتة")
            return self.prediction_cache[current_prime]
        
        # المرحلة 1: التقدير الأولي
        if verbose:
            print("📐 المرحلة 1: التقدير الأولي...")
        
        k_current = int(primepi(current_prime))
        k_next = k_current + 1
        
        prime_estimate = self._estimate_prime_from_zeta(k_next)
        
        if verbose:
            print(f"   ترتيب العدد الحالي: {k_current:,}")
            print(f"   تقدير العدد الأولي التالي: {prime_estimate:,}")
        
        # المرحلة 2: تحديد نافذة البحث
        if verbose:
            print("🎯 المرحلة 2: تحديد نافذة البحث...")
        
        window_start, window_end = self._adaptive_search_window(prime_estimate, current_prime)
        window_size = window_end - window_start
        
        if verbose:
            print(f"   نافذة البحث: [{window_start:,}, {window_end:,}]")
            print(f"   حجم النافذة: {window_size:,}")
        
        # المرحلة 3: الترشيح بـ GSE
        if verbose:
            print("🤖 المرحلة 3: الترشيح بـ GSE...")
        
        candidates = []
        gse_evaluations = 0
        
        # البحث في الأعداد الفردية فقط
        start_search = window_start if window_start % 2 == 1 else window_start + 1
        
        for x in range(start_search, window_end + 1, 2):
            if x <= current_prime:
                continue
            
            gse_prob = self._gse_prime_probability(x)
            gse_evaluations += 1
            
            if gse_prob >= gse_threshold:
                candidates.append((x, gse_prob))
            
            if len(candidates) >= max_candidates:
                break
        
        # ترتيب المرشحين حسب الاحتمالية
        candidates.sort(key=lambda x: x[1], reverse=True)
        
        if verbose:
            print(f"   تم تقييم {gse_evaluations:,} عدد")
            print(f"   تم العثور على {len(candidates)} مرشح قوي")
        
        # المرحلة 4: الاختبار الدقيق
        if verbose:
            print("⚡ المرحلة 4: الاختبار الدقيق...")
        
        primality_tests = 0
        next_prime = None
        
        # اختبار المرشحين
        for candidate, prob in candidates:
            primality_tests += 1
            
            if isprime(candidate):
                next_prime = candidate
                break
        
        # البحث الاحتياطي إذا لم نجد في المرشحين
        if next_prime is None:
            if verbose:
                print("   🔄 البحث الاحتياطي...")
            
            search_start = max(current_prime + 1, window_start)
            x = search_start if search_start % 2 == 1 else search_start + 1
            
            while x <= window_end + window_size:
                primality_tests += 1
                if isprime(x):
                    next_prime = x
                    break
                x += 2
        
        # حساب الإحصائيات
        total_time = time.time() - start_time
        
        if next_prime:
            gap = next_prime - current_prime
            estimate_error = abs(next_prime - prime_estimate)
            efficiency = gap / primality_tests if primality_tests > 0 else 0
            
            # تحديث الإحصائيات
            self.performance_stats['predictions'] += 1
            self.performance_stats['successes'] += 1
            self.performance_stats['total_time'] += total_time
            self.performance_stats['total_tests'] += primality_tests
            self.performance_stats['average_gap'] = (
                (self.performance_stats['average_gap'] * (self.performance_stats['successes'] - 1) + gap) /
                self.performance_stats['successes']
            )
            
            # حفظ في الذاكرة المؤقتة
            self.prediction_cache[current_prime] = next_prime
            
            if verbose:
                print(f"\n🎉 النتائج:")
                print(f"   العدد الأولي التالي: {next_prime:,}")
                print(f"   الفجوة: {gap}")
                print(f"   خطأ التقدير: {estimate_error:,}")
                print(f"   اختبارات الأولية: {primality_tests}")
                print(f"   الكفاءة: {efficiency:.2f}")
                print(f"   الوقت: {total_time:.4f} ثانية")
        else:
            self.performance_stats['predictions'] += 1
            if verbose:
                print("❌ فشل في العثور على العدد الأولي التالي")
        
        return next_prime
    
    def get_performance_stats(self) -> Dict[str, float]:
        """إرجاع إحصائيات الأداء"""
        stats = self.performance_stats.copy()
        
        if stats['predictions'] > 0:
            stats['success_rate'] = stats['successes'] / stats['predictions']
            stats['average_time'] = stats['total_time'] / stats['predictions']
            stats['average_tests'] = stats['total_tests'] / stats['predictions']
        
        return stats

# مثال للاستخدام
if __name__ == "__main__":
    # إنشاء نظام التنبؤ
    predictor = PrimePredictor()
    
    # اختبار التنبؤ
    test_prime = 1009
    next_prime = predictor.predict_next_prime(test_prime)
    
    if next_prime:
        print(f"\n✅ العدد الأولي التالي بعد {test_prime} هو {next_prime}")
        
        # التحقق من الصحة
        actual_next = nextprime(test_prime)
        if next_prime == actual_next:
            print("🎯 التنبؤ صحيح!")
        else:
            print(f"❌ التنبؤ خاطئ. العدد الصحيح هو {actual_next}")
    
    # عرض إحصائيات الأداء
    stats = predictor.get_performance_stats()
    print(f"\n📊 إحصائيات الأداء:")
    for key, value in stats.items():
        print(f"   {key}: {value}")
