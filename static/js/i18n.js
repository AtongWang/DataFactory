/**
 * WisdominDATA 国际化核心库
 * 支持多语言切换、文本替换和配置持久化
 */
class I18n {
    constructor() {
        this.currentLang = 'zh-CN'; // 默认中文
        this.languages = {}; // 语言包存储
        this.fallbackLang = 'zh-CN'; // 回退语言
        this.storageKey = 'wisdomindata_language'; // localStorage 键名
        
        // 初始化
        this.init();
    }
    
    /**
     * 初始化国际化系统
     */
    init() {
        // 从localStorage读取保存的语言设置
        const savedLang = localStorage.getItem(this.storageKey);
        if (savedLang && this.isValidLanguage(savedLang)) {
            this.currentLang = savedLang;
        }
        
        // 监听DOM变化，自动处理新添加的元素
        this.observeDOM();
    }
    
    /**
     * 验证语言代码是否有效
     */
    isValidLanguage(lang) {
        return ['zh-CN', 'en-US'].includes(lang);
    }
    
    /**
     * 注册语言包
     */
    registerLanguage(lang, translations) {
        this.languages[lang] = translations;
        console.log(`Language pack registered: ${lang}`);
    }
    
    /**
     * 切换语言
     */
    setLanguage(lang) {
        if (!this.isValidLanguage(lang)) {
            console.warn(`Unsupported language: ${lang}`);
            return false;
        }
        
        if (!this.languages[lang]) {
            console.warn(`Language pack not found: ${lang}`);
            return false;
        }
        
        this.currentLang = lang;
        
        // 保存到localStorage
        localStorage.setItem(this.storageKey, lang);
        
        // 更新页面文本
        this.updatePageText();
        
        // 触发语言切换事件
        this.trigger('languageChanged', { language: lang });
        
        return true;
    }
    
    /**
     * 获取当前语言
     */
    getCurrentLanguage() {
        return this.currentLang;
    }
    
    /**
     * 获取翻译文本
     */
    t(key, defaultText = '') {
        const lang = this.languages[this.currentLang];
        if (lang && lang[key]) {
            return lang[key];
        }
        
        // 尝试回退语言
        const fallbackLang = this.languages[this.fallbackLang];
        if (fallbackLang && fallbackLang[key]) {
            return fallbackLang[key];
        }
        
        // 返回默认文本或键名
        return defaultText || key;
    }
    
    /**
     * 更新页面中所有标记的文本
     */
    updatePageText() {
        // 更新带有 data-i18n 属性的元素
        $('[data-i18n]').each((index, element) => {
            const $el = $(element);
            const key = $el.data('i18n');
            const defaultText = $el.data('i18n-default') || $el.text().trim();
            
            if (key) {
                const translatedText = this.t(key, defaultText);
                $el.text(translatedText);
            }
        });
        
        // 更新带有 data-i18n-placeholder 属性的输入框
        $('[data-i18n-placeholder]').each((index, element) => {
            const $el = $(element);
            const key = $el.data('i18n-placeholder');
            const defaultText = $el.data('placeholder-default') || $el.attr('placeholder');
            
            if (key) {
                const translatedText = this.t(key, defaultText);
                $el.attr('placeholder', translatedText);
            }
        });
        
        // 更新带有 data-i18n-title 属性的元素
        $('[data-i18n-title]').each((index, element) => {
            const $el = $(element);
            const key = $el.data('i18n-title');
            const defaultText = $el.data('title-default') || $el.attr('title');
            
            if (key) {
                const translatedText = this.t(key, defaultText);
                $el.attr('title', translatedText);
            }
        });
    }
    
    /**
     * 监听DOM变化
     */
    observeDOM() {
        if (typeof MutationObserver !== 'undefined') {
            const observer = new MutationObserver((mutations) => {
                let shouldUpdate = false;
                mutations.forEach((mutation) => {
                    if (mutation.type === 'childList' && mutation.addedNodes.length > 0) {
                        // 检查新添加的节点是否包含需要翻译的元素
                        mutation.addedNodes.forEach((node) => {
                            if (node.nodeType === Node.ELEMENT_NODE) {
                                const $node = $(node);
                                if ($node.is('[data-i18n]') || $node.find('[data-i18n]').length > 0 ||
                                    $node.is('[data-i18n-placeholder]') || $node.find('[data-i18n-placeholder]').length > 0 ||
                                    $node.is('[data-i18n-title]') || $node.find('[data-i18n-title]').length > 0) {
                                    shouldUpdate = true;
                                }
                            }
                        });
                    }
                });
                
                if (shouldUpdate) {
                    // 延迟更新，避免频繁执行
                    clearTimeout(this.updateTimer);
                    this.updateTimer = setTimeout(() => {
                        this.updatePageText();
                    }, 100);
                }
            });
            
            observer.observe(document.body, {
                childList: true,
                subtree: true
            });
        }
    }
    
    /**
     * 事件系统
     */
    on(event, callback) {
        if (!this.events) this.events = {};
        if (!this.events[event]) this.events[event] = [];
        this.events[event].push(callback);
    }
    
    off(event, callback) {
        if (!this.events || !this.events[event]) return;
        const index = this.events[event].indexOf(callback);
        if (index > -1) {
            this.events[event].splice(index, 1);
        }
    }
    
    trigger(event, data) {
        if (!this.events || !this.events[event]) return;
        this.events[event].forEach(callback => {
            callback(data);
        });
    }
    
    /**
     * 获取可用语言列表
     */
    getAvailableLanguages() {
        return Object.keys(this.languages).map(lang => ({
            code: lang,
            name: this.getLanguageName(lang)
        }));
    }
    
    /**
     * 获取语言显示名称
     */
    getLanguageName(lang) {
        const names = {
            'zh-CN': '中文',
            'en-US': 'English'
        };
        return names[lang] || lang;
    }
    
    /**
     * 设置页面语言属性
     */
    setPageLanguage(lang) {
        document.documentElement.lang = lang;
    }
}

// 创建全局实例
window.i18n = new I18n();

// jQuery 插件扩展
if (typeof $ !== 'undefined') {
    $.fn.i18n = function() {
        return this.each(function() {
            const $el = $(this);
            const key = $el.data('i18n');
            if (key) {
                const defaultText = $el.data('i18n-default') || $el.text().trim();
                const translatedText = window.i18n.t(key, defaultText);
                $el.text(translatedText);
            }
        });
    };
}

// 导出模块（如果支持）
if (typeof module !== 'undefined' && module.exports) {
    module.exports = I18n;
} 