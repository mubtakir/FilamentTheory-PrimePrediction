#!/usr/bin/env python3
"""
المشغل الرئيسي لمشروع نظرية الفتائل
===================================

يشغل جميع مكونات المشروع من مكان واحد

تطوير: د. باسل يحيى عبدالله
"""

import os
import sys
from pathlib import Path

def main():
    """القائمة الرئيسية"""
    
    print("🌟" * 30)
    print("مشروع نظرية الفتائل - المشغل الرئيسي")
    print("د. باسل يحيى عبدالله")
    print("🌟" * 30)
    
    options = {
        "1": ("الصيغة الاختراقية النهائية", "01_CORE_THEORY/FINAL_BREAKTHROUGH_FORMULA.py"),
        "2": ("النظام المتكامل FilamentPrime", "02_FILAMENT_PRIME/FilamentPrime/run_demo.py"),
        "3": ("الخوارزمية المتقدمة", "03_ALGORITHMS/advanced_prime_algorithm.py"),
        "4": ("التحليل الرياضي", "03_ALGORITHMS/mathematical_analysis.py"),
        "5": ("عرض الفهرس الكامل", "MASTER_INDEX.md"),
        "0": ("خروج", None)
    }
    
    while True:
        print("\n📋 اختر ما تريد تشغيله:")
        for key, (desc, _) in options.items():
            print(f"   {key}. {desc}")
        
        choice = input("\n👉 اختيارك: ").strip()
        
        if choice == "0":
            print("🎉 شكراً لاستخدام نظرية الفتائل!")
            break
        elif choice in options:
            desc, file_path = options[choice]
            
            if choice == "5":
                # عرض الفهرس
                try:
                    with open("MASTER_INDEX.md", "r", encoding="utf-8") as f:
                        print("\n" + f.read())
                except FileNotFoundError:
                    print("❌ لم يتم العثور على الفهرس")
            else:
                # تشغيل الملف
                if file_path and Path(file_path).exists():
                    print(f"\n🚀 تشغيل: {desc}")
                    print("-" * 50)
                    os.system(f"python {file_path}")
                else:
                    print(f"❌ لم يتم العثور على: {file_path}")
        else:
            print("❌ اختيار غير صحيح")

if __name__ == "__main__":
    main()
