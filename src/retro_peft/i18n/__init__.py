"""
Internationalization (i18n) Support for Retro-PEFT-Adapters

Multi-language and multi-region support for global deployment
with localized error messages, documentation, and user interfaces.

Supported Languages:
- English (en) - Default
- Spanish (es)  
- French (fr)
- German (de)
- Japanese (ja)
- Chinese (zh)
"""

import json
import logging
import os
from pathlib import Path
from typing import Dict, Optional

logger = logging.getLogger(__name__)


class I18nManager:
    """Manages internationalization for the retro-peft library"""
    
    def __init__(self, default_locale: str = "en"):
        self.default_locale = default_locale
        self.current_locale = default_locale
        self.translations = {}
        self._load_translations()
        
    def _load_translations(self):
        """Load all translation files"""
        translations_dir = Path(__file__).parent / "translations"
        
        if not translations_dir.exists():
            logger.warning(f"Translations directory not found: {translations_dir}")
            return
            
        for locale_file in translations_dir.glob("*.json"):
            locale_code = locale_file.stem
            try:
                with open(locale_file, 'r', encoding='utf-8') as f:
                    self.translations[locale_code] = json.load(f)
                logger.debug(f"Loaded translations for locale: {locale_code}")
            except Exception as e:
                logger.error(f"Failed to load translations for {locale_code}: {e}")
                
    def set_locale(self, locale: str):
        """Set current locale"""
        if locale in self.translations or locale == self.default_locale:
            self.current_locale = locale
            logger.info(f"Locale set to: {locale}")
        else:
            logger.warning(f"Locale {locale} not supported, keeping {self.current_locale}")
            
    def get_text(self, key: str, **kwargs) -> str:
        """Get localized text by key"""
        # Try current locale first
        if self.current_locale in self.translations:
            text = self._get_nested_text(self.translations[self.current_locale], key)
            if text:
                return text.format(**kwargs) if kwargs else text
                
        # Fallback to default locale
        if (self.default_locale in self.translations and 
            self.current_locale != self.default_locale):
            text = self._get_nested_text(self.translations[self.default_locale], key)
            if text:
                return text.format(**kwargs) if kwargs else text
                
        # Fallback to key itself
        logger.warning(f"Translation not found for key: {key}")
        return key.replace("_", " ").title()
        
    def _get_nested_text(self, translations: Dict, key: str) -> Optional[str]:
        """Get nested translation using dot notation"""
        keys = key.split(".")
        current = translations
        
        for k in keys:
            if isinstance(current, dict) and k in current:
                current = current[k]
            else:
                return None
                
        return current if isinstance(current, str) else None
        
    def get_supported_locales(self) -> list:
        """Get list of supported locales"""
        return list(self.translations.keys())


# Global i18n manager instance
_i18n_manager = I18nManager()


def set_locale(locale: str):
    """Set global locale"""
    _i18n_manager.set_locale(locale)
    

def get_text(key: str, **kwargs) -> str:
    """Get localized text"""
    return _i18n_manager.get_text(key, **kwargs)
    

def get_supported_locales() -> list:
    """Get supported locales"""
    return _i18n_manager.get_supported_locales()


# Convenience functions for common text categories
def error_msg(key: str, **kwargs) -> str:
    """Get error message"""
    return get_text(f"errors.{key}", **kwargs)
    

def info_msg(key: str, **kwargs) -> str:
    """Get info message"""
    return get_text(f"info.{key}", **kwargs)
    

def warning_msg(key: str, **kwargs) -> str:
    """Get warning message"""
    return get_text(f"warnings.{key}", **kwargs)
    

def ui_text(key: str, **kwargs) -> str:
    """Get UI text"""
    return get_text(f"ui.{key}", **kwargs)