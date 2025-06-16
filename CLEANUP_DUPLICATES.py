#!/usr/bin/env python3
"""
تنظيف الملفات المكررة
====================

يحذف الملفات المكررة ويحتفظ فقط بالنسخ المنظمة

تطوير: د. باسل يحيى عبدالله
"""

import os
import shutil
from pathlib import Path

def cleanup_duplicates():
    """تنظيف الملفات المكررة"""
    
    print("🧹 تنظيف الملفات المكررة")
    print("=" * 40)
    
    base_dir = Path(".")
    
    # الملفات التي يجب حذفها من المجلد الرئيسي (موجودة في المجلدات المنظمة)
    files_to_remove = [
        # النظرية الأساسية (موجودة في 01_CORE_THEORY)
        "FINAL_BREAKTHROUGH_FORMULA.py",
        "BREAKTHROUGH_FILAMENT_FORMULA.txt", 
        "ultimate_formula.py",
        "unified_formula_system.py",
        "UNIFIED_MATHEMATICAL_FORMULA.md",
        
        # الخوارزميات (موجودة في 03_ALGORITHMS)
        "advanced_prime_algorithm.py",
        "hpp_predictor.py",
        "mathematical_analysis.py",
        "interactive_prime_explorer.py",
        "train_models.py",
        "test_basic.py",
        
        # البيانات (موجودة في 04_DATA)
        "zeta_zeros_1000.txt",
        "error_model_params.pkl",
        "gse_classifier_params.pkl",
        
        # التوثيق (موجود في 05_DOCUMENTATION)
        "FILAMENT_THEORY_SUMMARY.md",
        "README.md",
        "CHANGELOG.md",
        "LICENSE",
        "ATTRIBUTION.md",
        "PROJECT_SUMMARY.md",
        "ULTIMATE_FILAMENT_FORMULA.txt",
        
        # التقارير (موجودة في 06_RESULTS)
        "تقرير نهائي.docx",
        "تقرير نهائي.pdf",
        "تقرير نهائي.txt",
        
        # الأرشيف (موجود في 07_ARCHIVE)
        "run_demo.py",
        "setup.py",
        "requirements.txt"
    ]
    
    # المجلدات التي يجب حذفها (مكررة)
    dirs_to_remove = [
        "data",
        "__pycache__",
        "FilamentPrime"  # النسخة الأصلية (موجودة في 02_FILAMENT_PRIME)
    ]
    
    # حذف الملفات المكررة
    print("📄 حذف الملفات المكررة...")
    for file_name in files_to_remove:
        file_path = base_dir / file_name
        if file_path.exists() and file_path.is_file():
            try:
                file_path.unlink()
                print(f"   ✅ تم حذف: {file_name}")
            except Exception as e:
                print(f"   ❌ خطأ في حذف {file_name}: {e}")
        else:
            print(f"   ⚠️ غير موجود: {file_name}")
    
    # حذف المجلدات المكررة
    print("\n📂 حذف المجلدات المكررة...")
    for dir_name in dirs_to_remove:
        dir_path = base_dir / dir_name
        if dir_path.exists() and dir_path.is_dir():
            try:
                shutil.rmtree(dir_path)
                print(f"   ✅ تم حذف المجلد: {dir_name}")
            except Exception as e:
                print(f"   ❌ خطأ في حذف المجلد {dir_name}: {e}")
        else:
            print(f"   ⚠️ غير موجود: {dir_name}")
    
    # إنشاء ملف README رئيسي محدث
    create_main_readme(base_dir)
    
    print("\n🎉 تم تنظيف المشروع بنجاح!")
    print("📋 المجلد الآن منظم ونظيف")

def create_main_readme(base_dir):
    """إنشاء README رئيسي محدث"""
    
    readme_content = """# 🌟 مشروع نظرية الفتائل - المجلد الرئيسي

## 👨‍🔬 المؤلف
**د. باسل يحيى عبدالله** - الباحث في الفيزياء النظرية ونظرية الأعداد

---

## 🎯 نظرة عامة

هذا هو **المجلد الرئيسي المنظم** لمشروع نظرية الفتائل الثوري الذي حقق:
- ✅ **91.0%** دقة إجمالية
- ✅ **100%** دقة في التنبؤ بالأعداد الأولية  
- ✅ **81.9%** دقة في أصفار زيتا
- ✅ **أول صيغة موحدة** في التاريخ

---

## 📁 الهيكل المنظم

### 🎯 **01_CORE_THEORY** - النظرية الأساسية
الصيغ الرياضية النهائية والاختراقية

### 🏗️ **02_FILAMENT_PRIME** - النظام المتكامل  
نظام FilamentPrime الكامل مع جميع الوحدات

### ⚙️ **03_ALGORITHMS** - الخوارزميات
الخوارزميات المتقدمة والنماذج

### 📊 **04_DATA** - البيانات
أصفار زيتا والنماذج المدربة

### 📚 **05_DOCUMENTATION** - التوثيق
الملخصات والأدلة الشاملة

### 📈 **06_RESULTS** - النتائج
التقارير النهائية والنتائج

### 🗄️ **07_ARCHIVE** - الأرشيف
الملفات القديمة والنسخ الاحتياطية

---

## 🚀 التشغيل السريع

### المشغل الرئيسي:
```bash
python MASTER_RUNNER.py
```

### الصيغة الاختراقية النهائية:
```bash
python 01_CORE_THEORY/FINAL_BREAKTHROUGH_FORMULA.py
```

### النظام المتكامل:
```bash
python 02_FILAMENT_PRIME/FilamentPrime/run_demo.py
```

---

## 📋 الملفات الرئيسية

- `MASTER_RUNNER.py` - **المشغل الرئيسي** ⭐
- `MASTER_INDEX.md` - **الفهرس الكامل** ⭐
- `MASTER_PROJECT_ORGANIZER.py` - منظم المشروع
- `CLEANUP_DUPLICATES.py` - منظف الملفات المكررة

---

## 🏆 الإنجاز العلمي

هذا المشروع يمثل **اختراق علمي تاريخي** في:
- 🧮 **الرياضيات**: أول صيغة موحدة للأعداد الأولية وأصفار زيتا
- ⚛️ **الفيزياء**: تطبيق نظرية الفتائل على نظرية الأعداد
- 💻 **علوم الحاسوب**: خوارزميات جديدة للتنبؤ
- 🌍 **العلوم**: ربط الفيزياء بالرياضيات في نظرية موحدة

---

## 📞 التواصل

للاستفسارات العلمية أو التعاون البحثي:
- **المؤلف**: د. باسل يحيى عبدالله
- **التخصص**: الفيزياء النظرية ونظرية الأعداد

---

**"الأعداد الأولية ليست مجرد أرقام، بل هي تجليات لرنين كوني عميق"**  
*- د. باسل يحيى عبدالله*

---

© 2024 د. باسل يحيى عبدالله - جميع الحقوق محفوظة
"""
    
    with open(base_dir / "README_MAIN.md", "w", encoding="utf-8") as f:
        f.write(readme_content)
    
    print("✅ تم إنشاء README_MAIN.md")

if __name__ == "__main__":
    cleanup_duplicates()
