"""
نظرية الفتائل - الأسس الفيزيائية والرياضية
==========================================

تطوير: د. باسل يحيى عبدالله

هذه الوحدة تطبق المبادئ الأساسية لنظرية الفتائل:
- ديناميكية الصفر والازدواجية المتعامدة
- الرنين الكوني والثوابت الفيزيائية
- التوازن الكوني ومعادلات الاستقرار
"""

import numpy as np
from scipy.constants import h, c, epsilon_0, mu_0
from typing import Tuple, Dict, Any, List
import logging

class FilamentTheory:
    """
    تطبيق نظرية الفتائل الأساسية
    
    المبادئ الأساسية:
    1. الصفر ينبثق إلى ضدين متعامدين (تكتل/اتساع)
    2. الرنين كقانون أساسي للتكوين
    3. التوازن الكوني والاستقرار
    """
    
    def __init__(self):
        """تهيئة الثوابت الأساسية"""
        # الثوابت الفيزيائية
        self.h = h  # ثابت بلانك
        self.c = c  # سرعة الضوء
        self.epsilon_0 = epsilon_0  # نفاذية الفراغ الكهربائية
        self.mu_0 = mu_0  # نفاذية الفراغ المغناطيسية
        
        # الثوابت المشتقة من النظرية
        self.f_0 = 1 / (4 * np.pi)  # التردد الأساسي
        self.E_0 = self.h * self.f_0  # الطاقة الأساسية
        self.m_0 = self.E_0 / (self.c ** 2)  # الكتلة الأساسية للفتيلة
        self.Z_0 = np.sqrt(self.mu_0 / self.epsilon_0)  # الممانعة المميزة
        
        # معاملات التناظر الثلاثي
        self.mass_capacitance_ratio = self.m_0 / self.epsilon_0
        self.space_inductance_ratio = 1 / (self.c * self.mu_0)
        
        logging.info("تم تهيئة نظرية الفتائل بنجاح")
        self._log_constants()
    
    def _log_constants(self):
        """طباعة الثوابت الأساسية"""
        print("🌌 ثوابت نظرية الفتائل:")
        print(f"   التردد الأساسي f₀ = {self.f_0:.6e} Hz")
        print(f"   الطاقة الأساسية E₀ = {self.E_0:.6e} J")
        print(f"   الكتلة الأساسية m₀ = {self.m_0:.6e} kg")
        print(f"   الممانعة المميزة Z₀ = {self.Z_0:.2f} Ω")
    
    def zero_dynamics(self, n: int) -> Dict[str, float]:
        """
        حساب ديناميكية الصفر للحالة n
        
        Args:
            n: رقم الحالة (العدد الأولي أو ترتيبه)
            
        Returns:
            قاموس يحتوي على خصائص الحالة
        """
        # الطاقة الكلية للحالة
        total_energy = n * self.E_0
        
        # الكتلة المكافئة
        equivalent_mass = n * self.m_0
        
        # التردد المكافئ
        equivalent_frequency = total_energy / self.h
        
        # الطاقة التكتلية (الجزء الحقيقي)
        aggregative_energy = np.log(n) * self.E_0
        
        # الطاقة الاتساعية (الجزء التخيلي)
        expansive_energy = self.E_0 / np.sqrt(n)
        
        return {
            'total_energy': total_energy,
            'equivalent_mass': equivalent_mass,
            'equivalent_frequency': equivalent_frequency,
            'aggregative_energy': aggregative_energy,
            'expansive_energy': expansive_energy,
            'resonance_condition': self._check_resonance(n)
        }
    
    def _check_resonance(self, n: int) -> bool:
        """فحص شرط الرنين للحالة n"""
        # شرط الرنين: تساوي الممانعة الحثية والسعوية
        inductive_impedance = 2 * np.pi * self.f_0 * n * self.mu_0
        capacitive_impedance = 1 / (2 * np.pi * self.f_0 * n * self.epsilon_0)
        
        # التحقق من التقارب (ضمن هامش خطأ)
        ratio = inductive_impedance / capacitive_impedance
        return abs(ratio - 1) < 0.1  # هامش خطأ 10%
    
    def orthogonal_duality(self, state_vector: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        تطبيق مبدأ الازدواجية المتعامدة
        
        Args:
            state_vector: متجه الحالة
            
        Returns:
            (الجزء التكتلي، الجزء الاتساعي)
        """
        # تحليل المتجه إلى مكونين متعامدين
        aggregative_component = np.real(state_vector)
        expansive_component = 1j * np.imag(state_vector)
        
        # التحقق من التعامد
        dot_product = np.dot(aggregative_component, np.imag(expansive_component))
        
        if abs(dot_product) > 1e-10:
            logging.warning("تحذير: المكونان ليسا متعامدين تماماً")
        
        return aggregative_component, expansive_component
    
    def cosmic_balance_equation(self, n: int) -> float:
        """
        معادلة التوازن الكوني للحالة n
        
        Returns:
            قيمة التوازن (يجب أن تكون قريبة من الصفر)
        """
        dynamics = self.zero_dynamics(n)
        
        # معادلة التوازن: الطاقة التكتلية - الطاقة الاتساعية = 0
        balance = (dynamics['aggregative_energy'] - 
                  dynamics['expansive_energy'])
        
        return balance / self.E_0  # تسوية بالطاقة الأساسية
    
    def predict_stable_states(self, max_n: int = 1000) -> np.ndarray:
        """
        التنبؤ بالحالات المستقرة (الأعداد الأولية المحتملة)
        
        Args:
            max_n: أقصى قيمة للبحث
            
        Returns:
            مصفوفة الحالات المستقرة
        """
        stable_states = []
        
        for n in range(2, max_n + 1):
            balance = abs(self.cosmic_balance_equation(n))
            resonance = self._check_resonance(n)
            
            # شروط الاستقرار
            if balance < 0.5 and resonance:
                stable_states.append(n)
        
        return np.array(stable_states)

# مثال للاستخدام
if __name__ == "__main__":
    # إنشاء نموذج النظرية
    theory = FilamentTheory()
    
    # اختبار ديناميكية الصفر
    dynamics = theory.zero_dynamics(17)  # للعدد الأولي 17
    print(f"\n🔬 ديناميكية الصفر للعدد 17:")
    for key, value in dynamics.items():
        print(f"   {key}: {value}")
    
    # التنبؤ بالحالات المستقرة
    stable = theory.predict_stable_states(100)
    print(f"\n🎯 الحالات المستقرة المتوقعة (أول 20): {stable[:20]}")
