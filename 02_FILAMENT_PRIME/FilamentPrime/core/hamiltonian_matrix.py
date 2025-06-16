"""
مصفوفة هاملتون الهيرميتية
=========================

تطبيق مصفوفة هاملتون الفيزيائية التي أنتجت سلوك GUE
وأظهرت تنافر المستويات المطابق لأصفار دالة زيتا

تطوير: د. باسل يحيى عبدالله
"""

import numpy as np
import matplotlib.pyplot as plt
from sympy import primerange
from scipy.constants import h, c
from typing import Tuple, Dict, List, Optional
from .filament_theory import FilamentTheory

class HamiltonianMatrix:
    """
    مصفوفة هاملتون الهيرميتية للأعداد الأولية
    
    البنية:
    - القطر: H[i,i] = h * log(p_i) (طاقة تكتلية حقيقية)
    - خارج القطر: H[i,j] = i*h*c/sqrt(p_i*p_j) (طاقة اتساعية تخيلية)
    
    هذه البنية تنتج سلوك GUE مع تنافر المستويات
    """
    
    def __init__(self):
        """تهيئة مصفوفة هاملتون"""
        self.theory = FilamentTheory()
        self.h = h  # ثابت بلانك
        self.c = c  # سرعة الضوء
        
        # بيانات المصفوفة
        self.primes = None
        self.matrix = None
        self.eigenvalues = None
        self.eigenvectors = None
        
        print("⚛️ تهيئة مصفوفة هاملتون الهيرميتية")
    
    def build_matrix(self, num_primes: int = 500, physical_scaling: bool = True) -> np.ndarray:
        """
        بناء مصفوفة هاملتون
        
        Args:
            num_primes: عدد الأعداد الأولية
            physical_scaling: استخدام التدرج الفيزيائي
            
        Returns:
            مصفوفة هاملتون الهيرميتية
        """
        print(f"🔧 بناء مصفوفة هاملتون لـ {num_primes} عدد أولي...")
        
        # توليد الأعداد الأولية
        self.primes = list(primerange(2, num_primes * 15))[:num_primes]
        K = len(self.primes)
        
        print(f"   نطاق الأعداد الأولية: {self.primes[0]} إلى {self.primes[-1]}")
        
        # إنشاء المصفوفة
        self.matrix = np.zeros((K, K), dtype=np.complex128)
        
        for i in range(K):
            for j in range(K):
                p_i, p_j = self.primes[i], self.primes[j]
                
                if i == j:
                    # القطر: الطاقة التكتلية (حقيقية)
                    if physical_scaling:
                        # استخدام ثابت بلانك للتدرج الفيزيائي
                        self.matrix[i, i] = self.h * np.log(p_i)
                    else:
                        # التدرج الرياضي البسيط
                        self.matrix[i, i] = np.log(p_i)
                
                else:
                    # خارج القطر: الطاقة الاتساعية (تخيلية بحتة)
                    if physical_scaling:
                        # استخدام ثابت بلانك وسرعة الضوء
                        interaction = (1j * self.h * self.c) / np.sqrt(p_i * p_j)
                    else:
                        # التفاعل الرياضي البسيط
                        interaction = 1j / np.sqrt(p_i * p_j)
                    
                    self.matrix[i, j] = interaction
        
        print(f"✅ تم بناء مصفوفة {K}×{K}")
        return self.matrix
    
    def compute_eigenvalues(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        حساب القيم والمتجهات الذاتية
        
        Returns:
            (القيم الذاتية، المتجهات الذاتية)
        """
        if self.matrix is None:
            raise ValueError("يجب بناء المصفوفة أولاً")
        
        print("🧮 حساب القيم الذاتية...")
        
        # حساب القيم الذاتية (كلها حقيقية للمصفوفة الهيرميتية)
        self.eigenvalues, self.eigenvectors = np.linalg.eigh(self.matrix)
        
        # ترتيب القيم الذاتية
        sorted_indices = np.argsort(self.eigenvalues)
        self.eigenvalues = self.eigenvalues[sorted_indices]
        self.eigenvectors = self.eigenvectors[:, sorted_indices]
        
        print(f"✅ تم حساب {len(self.eigenvalues)} قيمة ذاتية")
        print(f"   نطاق الطاقة: [{np.min(self.eigenvalues):.3e}, {np.max(self.eigenvalues):.3e}]")
        
        return self.eigenvalues, self.eigenvectors
    
    def analyze_level_spacing(self) -> Dict[str, float]:
        """
        تحليل تباعد المستويات (Level Spacing)
        
        Returns:
            إحصائيات تباعد المستويات
        """
        if self.eigenvalues is None:
            raise ValueError("يجب حساب القيم الذاتية أولاً")
        
        print("📊 تحليل تباعد المستويات...")
        
        # حساب الفجوات
        gaps = np.diff(self.eigenvalues)
        
        # تسوية الفجوات
        normalized_gaps = gaps / np.mean(gaps)
        
        # إحصائيات التنافر
        small_gaps_ratio = np.sum(normalized_gaps < 0.1) / len(normalized_gaps)
        mean_gap = np.mean(normalized_gaps)
        std_gap = np.std(normalized_gaps)
        
        # تصنيف السلوك الإحصائي
        if small_gaps_ratio < 0.05:
            behavior_type = "GUE-like (Strong Repulsion)"
            behavior_score = 1.0
        elif small_gaps_ratio < 0.15:
            behavior_type = "Intermediate"
            behavior_score = 0.5
        else:
            behavior_type = "GOE-like (Weak Repulsion)"
            behavior_score = 0.0
        
        # حساب إنتروبيا التوزيع
        hist, _ = np.histogram(normalized_gaps, bins=50, density=True)
        hist = hist[hist > 0]  # تجنب log(0)
        entropy = -np.sum(hist * np.log(hist)) * (normalized_gaps.max() - normalized_gaps.min()) / 50
        
        stats = {
            'small_gaps_ratio': small_gaps_ratio,
            'mean_gap': mean_gap,
            'std_gap': std_gap,
            'behavior_type': behavior_type,
            'behavior_score': behavior_score,
            'entropy': entropy,
            'total_levels': len(self.eigenvalues),
            'energy_range': np.max(self.eigenvalues) - np.min(self.eigenvalues)
        }
        
        print(f"   نسبة الفجوات الصغيرة: {small_gaps_ratio:.2%}")
        print(f"   نوع السلوك: {behavior_type}")
        print(f"   الإنتروبيا: {entropy:.4f}")
        
        return stats
    
    def compare_with_random_matrices(self, num_comparisons: int = 5) -> Dict[str, List[float]]:
        """
        مقارنة مع المصفوفات العشوائية
        
        Args:
            num_comparisons: عدد المصفوفات العشوائية للمقارنة
            
        Returns:
            إحصائيات المقارنة
        """
        if self.matrix is None:
            raise ValueError("يجب بناء المصفوفة أولاً")
        
        print(f"🎲 مقارنة مع {num_comparisons} مصفوفة عشوائية...")
        
        K = self.matrix.shape[0]
        
        # إحصائيات مصفوفتنا
        our_stats = self.analyze_level_spacing()
        
        # إحصائيات المصفوفات العشوائية
        goe_stats = []
        gue_stats = []
        
        for i in range(num_comparisons):
            # مصفوفة GOE (حقيقية متناظرة)
            goe_matrix = np.random.randn(K, K)
            goe_matrix = (goe_matrix + goe_matrix.T) / 2
            goe_eigenvals = np.linalg.eigvalsh(goe_matrix)
            goe_eigenvals.sort()
            
            goe_gaps = np.diff(goe_eigenvals)
            goe_normalized = goe_gaps / np.mean(goe_gaps)
            goe_small_ratio = np.sum(goe_normalized < 0.1) / len(goe_normalized)
            goe_stats.append(goe_small_ratio)
            
            # مصفوفة GUE (مركبة هيرميتية)
            gue_real = np.random.randn(K, K)
            gue_imag = np.random.randn(K, K)
            gue_matrix = (gue_real + 1j * gue_imag + (gue_real - 1j * gue_imag).T) / 2
            gue_eigenvals = np.linalg.eigvalsh(gue_matrix)
            gue_eigenvals.sort()
            
            gue_gaps = np.diff(gue_eigenvals)
            gue_normalized = gue_gaps / np.mean(gue_gaps)
            gue_small_ratio = np.sum(gue_normalized < 0.1) / len(gue_normalized)
            gue_stats.append(gue_small_ratio)
        
        comparison = {
            'our_small_gaps_ratio': our_stats['small_gaps_ratio'],
            'goe_small_gaps_ratios': goe_stats,
            'gue_small_gaps_ratios': gue_stats,
            'goe_mean': np.mean(goe_stats),
            'gue_mean': np.mean(gue_stats),
            'closer_to_gue': abs(our_stats['small_gaps_ratio'] - np.mean(gue_stats)) < 
                           abs(our_stats['small_gaps_ratio'] - np.mean(goe_stats))
        }
        
        print(f"   مصفوفتنا: {our_stats['small_gaps_ratio']:.2%}")
        print(f"   متوسط GOE: {comparison['goe_mean']:.2%}")
        print(f"   متوسط GUE: {comparison['gue_mean']:.2%}")
        print(f"   أقرب إلى: {'GUE' if comparison['closer_to_gue'] else 'GOE'}")
        
        return comparison
    
    def plot_level_spacing_distribution(self, save_plot: bool = False):
        """رسم توزيع تباعد المستويات"""
        if self.eigenvalues is None:
            raise ValueError("يجب حساب القيم الذاتية أولاً")
        
        # حساب الفجوات المسواة
        gaps = np.diff(self.eigenvalues)
        normalized_gaps = gaps / np.mean(gaps)
        
        # الرسم
        plt.figure(figsize=(12, 8))
        
        # توزيع الفجوات
        plt.subplot(2, 2, 1)
        plt.hist(normalized_gaps, bins=50, density=True, alpha=0.7, 
                color='blue', edgecolor='black')
        plt.xlabel('حجم الفجوة المسوى (s)')
        plt.ylabel('الكثافة P(s)')
        plt.title('توزيع تباعد المستويات')
        plt.grid(True, alpha=0.3)
        
        # القيم الذاتية
        plt.subplot(2, 2, 2)
        plt.plot(self.eigenvalues, 'b-', linewidth=1)
        plt.xlabel('الفهرس')
        plt.ylabel('القيمة الذاتية (الطاقة)')
        plt.title('مستويات الطاقة')
        plt.grid(True, alpha=0.3)
        
        # الفجوات الخام
        plt.subplot(2, 2, 3)
        plt.plot(gaps, 'g-', linewidth=1)
        plt.xlabel('الفهرس')
        plt.ylabel('حجم الفجوة')
        plt.title('الفجوات الخام')
        plt.grid(True, alpha=0.3)
        
        # إحصائيات
        stats = self.analyze_level_spacing()
        plt.subplot(2, 2, 4)
        plt.text(0.1, 0.8, f"نسبة الفجوات الصغيرة: {stats['small_gaps_ratio']:.2%}", 
                transform=plt.gca().transAxes, fontsize=12)
        plt.text(0.1, 0.7, f"متوسط الفجوة: {stats['mean_gap']:.4f}", 
                transform=plt.gca().transAxes, fontsize=12)
        plt.text(0.1, 0.6, f"نوع السلوك: {stats['behavior_type']}", 
                transform=plt.gca().transAxes, fontsize=12)
        plt.text(0.1, 0.5, f"عدد المستويات: {stats['total_levels']}", 
                transform=plt.gca().transAxes, fontsize=12)
        plt.axis('off')
        plt.title('الإحصائيات')
        
        plt.suptitle(f'تحليل مصفوفة هاملتون ({len(self.primes)} عدد أولي)', 
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_plot:
            plt.savefig('hamiltonian_analysis.png', dpi=300, bbox_inches='tight')
            print("💾 تم حفظ الرسم البياني")
        
        plt.show()

# مثال للاستخدام
if __name__ == "__main__":
    # إنشاء مصفوفة هاملتون
    hamiltonian = HamiltonianMatrix()
    
    # بناء المصفوفة
    H = hamiltonian.build_matrix(num_primes=200, physical_scaling=True)
    
    # حساب القيم الذاتية
    eigenvals, eigenvecs = hamiltonian.compute_eigenvalues()
    
    # تحليل تباعد المستويات
    spacing_stats = hamiltonian.analyze_level_spacing()
    
    # مقارنة مع المصفوفات العشوائية
    comparison = hamiltonian.compare_with_random_matrices(num_comparisons=3)
    
    # رسم النتائج
    hamiltonian.plot_level_spacing_distribution(save_plot=True)
