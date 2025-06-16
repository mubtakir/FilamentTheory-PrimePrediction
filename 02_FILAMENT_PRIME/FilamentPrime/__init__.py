"""
FilamentPrime: نظام التنبؤ المتكامل للأعداد الأولية
=================================================

تطبيق شامل لنظرية الفتائل للدكتور باسل يحيى عبدالله
لفهم والتنبؤ بالأعداد الأولية وأصفار دالة زيتا.

المؤلف: د. باسل يحيى عبدالله
الإصدار: 1.0.0
التاريخ: 2024

الوحدات الرئيسية:
- filament_theory: نظرية الفتائل الأساسية
- zeta_predictor: التنبؤ بأصفار دالة زيتا
- prime_predictor: التنبؤ بالأعداد الأولية
- gse_model: نموذج GSE المحسن
- hamiltonian_matrix: مصفوفة هاملتون الهيرميتية
"""

__version__ = "1.0.0"
__author__ = "د. باسل يحيى عبدالله"
__email__ = "basel.yahya@example.com"
__description__ = "نظام التنبؤ المتكامل للأعداد الأولية باستخدام نظرية الفتائل"

# استيراد الوحدات الرئيسية
try:
    from .core.filament_theory import FilamentTheory
    from .core.zeta_predictor import ZetaZerosPredictor
    from .core.prime_predictor import PrimePredictor
    from .core.gse_model import GSEModel
    from .core.hamiltonian_matrix import HamiltonianMatrix
    
    __all__ = [
        'FilamentTheory',
        'ZetaZerosPredictor', 
        'PrimePredictor',
        'GSEModel',
        'HamiltonianMatrix'
    ]
    
except ImportError as e:
    print(f"تحذير: لم يتم تحميل بعض الوحدات: {e}")
    __all__ = []

def get_version():
    """إرجاع إصدار المكتبة"""
    return __version__

def get_info():
    """إرجاع معلومات المكتبة"""
    return {
        'name': 'FilamentPrime',
        'version': __version__,
        'author': __author__,
        'description': __description__,
        'modules': __all__
    }

# رسالة ترحيب
print("🌟 مرحباً بك في FilamentPrime!")
print(f"📚 نظرية الفتائل للدكتور {__author__}")
print(f"🔬 الإصدار: {__version__}")
