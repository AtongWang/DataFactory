/**
 * i18n Helper for Dynamic Content Translation
 * 处理动态生成内容的国际化辅助函数
 */

// 等待window.i18n加载完成的辅助函数
function waitForI18n() {
    return new Promise((resolve) => {
        if (window.i18n && window.i18n.t) {
            resolve();
        } else {
            setTimeout(() => waitForI18n().then(resolve), 100);
        }
    });
}

// 翻译文本的辅助函数
function t(key, options = {}) {
    let text = '';
    
    if (window.i18n && window.i18n.t) {
        text = window.i18n.t(key);
    } else {
        // 如果i18n未加载，返回中文默认值或键名
        const fallbacks = {
            'common.loading': '加载中...',
            'js.please_select_file': '请选择文件',
            'js.please_select_target_table': '追加模式下，请先选择目标表',
            'js.please_select_table_to_append': '请选择要追加数据的目标表',
            'js.get_table_list_failed': '获取现有表列表失败',
            'js.get_table_schema_failed': '获取表结构失败',
            'js.network_error': '网络错误',
            'js.unknown_error': '未知错误',
            'js.data_import_success': '数据导入成功！',
            'js.data_import_failed': '数据导入失败',
            'js.importing_data': '正在导入数据...',
            'js.loading_tables': '-- 加载中... --',
            'js.load_failed': '-- 加载失败 --',
            'js.no_available_tables': '-- 无可用数据表 --',
            'js.cannot_load_tables': '-- 无法加载数据表 --',
            'js.load_tables_failed': '-- 加载数据表失败 --',
            'js.get_data_failed': '获取数据失败',
            'js.checking_connection': '正在检查连接...',
            'js.loading_graphs': '-- 正在加载... --',
            'js.no_available_graphs': '-- 无可用知识图谱 --',
            'js.load_graphs_failed': '-- 加载知识图谱失败 --',
            'js.please_select_table_first': '请先选择数据表',
            'js.table_locked': '数据表已锁定',
            'js.table_unlocked': '数据表已解锁',
            'js.loading_data': '正在加载数据...',
            'js.loading_knowledge_data': '正在加载知识数据...',
            'js.loading_data_description': '正在加载数据描述...',
            'js.loading_sql_qa_data': '正在加载SQL问答对数据...',
            'js.loading_cypher_qa_data': '正在加载Cypher问答对数据...',
            'js.loading_graph_data': '正在加载图谱数据...',
            'js.generating_ddl': '正在生成 DDL...',
            'js.ai_thinking': 'AI 正在思考...',
            'js.ai_generating_descriptions': 'AI 正在为所有列生成中文描述...',
            'js.saving_language_settings': '正在保存语言设置...',
            'js.get_models_failed': '获取模型列表失败',
            'js.check_network_connection': '获取模型列表失败，请检查网络连接',
            'js.entity_type_select_identifier': '实体类型 "{name}": 请选择一个标识符列',
            'js.relationship_select_required': '关系 {index}: 请选择源实体、目标实体并输入关系类型名称',
            'js.rule_condition_select_required': '规则条件: 请选择列和操作符',
            'js.semantic_condition_select_required': '语义条件: 请选择至少一个用于计算的列',
            'js.inter_entity_comparison_required': '实体间比较条件: 请选择源列、目标列和操作符',
            'js.please_go_to_database_management': '请到"数据管理"页面添加或更新数据表描述 (DDL)。',
            'js.please_select_graph_for_visualization': '请选择一个知识图谱进行可视化',
            'js.get_graph_list_failed': '获取知识图谱列表失败',
            'js.get_metrics_failed': '获取指标失败',
            'js.get_analysis_report_failed': '获取分析报告失败',
            'js.get_database_config_failed': '获取数据库配置失败',
            'js.get_database_tables_failed': '获取数据库表失败',
            'js.get_table_structure_failed': '获取表结构失败',
            'js.get_knowledge_graph_list_failed': '获取知识图谱列表失败',
            'js.starting_graph_construction': '正在开始构建知识图谱...',
            'js.select_data_table': '-- 请选择数据表 --',
            'js.select_knowledge_graph': '-- 请选择知识图谱 --',
            'js.select_target_table': '请选择目标表',
            'js.table_description': '数据表描述 (ID: {id})',
            'js.table_with_name': '数据表: {name} (ID: {id})',
            'js.table_locked_status': '数据表已锁定',
            'js.table_unlocked_status': '数据表未锁定'
        };
        
        text = fallbacks[key] || key;
    }
    
    // 处理模板变量替换（在所有情况下都执行）
    if (options && typeof options === 'object') {
        Object.keys(options).forEach(optionKey => {
            const placeholder = `{${optionKey}}`;
            text = text.replace(new RegExp(placeholder.replace(/[{}]/g, '\\$&'), 'g'), options[optionKey]);
        });
    }
    
    return text;
}

// 动态创建带翻译的HTML选项
function createTranslatedOption(value, textKey, defaultText, isSelected = false) {
    const selected = isSelected ? 'selected' : '';
    const translatedText = t(textKey, defaultText);
    return `<option value="${value}" ${selected}>${translatedText}</option>`;
}

// 动态创建带翻译的HTML内容
function createTranslatedHTML(tag, textKey, defaultText, classes = '', attributes = '') {
    const translatedText = t(textKey, defaultText);
    return `<${tag} class="${classes}" ${attributes}>${translatedText}</${tag}>`;
}

// 更新现有元素的文本内容
function updateElementText(selector, textKey, options = {}) {
    const element = $(selector);
    if (element.length > 0) {
        element.text(t(textKey, options));
    }
}

// 显示翻译后的alert消息
function showTranslatedAlert(textKey, options = {}) {
    alert(t(textKey, options));
}

// 显示翻译后的错误消息
function showTranslatedError(textKey, options = {}) {
    const message = t(textKey, options);
    console.error(message);
    // 如果有通用的错误显示函数，可以在这里调用
    if (typeof showError === 'function') {
        showError(message);
    } else {
        alert(message);
    }
}

// 显示翻译后的成功消息
function showTranslatedSuccess(textKey, options = {}) {
    const message = t(textKey, options);
    console.log(message);
    // 如果有通用的成功显示函数，可以在这里调用
    if (typeof showSuccess === 'function') {
        showSuccess(message);
    } else {
        alert(message);
    }
}

// 创建加载状态的HTML
function createLoadingHTML(textKey = 'common.loading') {
    return `<div class="text-center">
        <div class="spinner-border" role="status">
            <span class="visually-hidden">${t(textKey)}</span>
        </div>
    </div>`;
}

// 创建带翻译的加载消息
function createLoadingMessage(textKey = 'common.loading') {
    return `<div class="text-center">
        <div class="spinner-border spinner-border-sm me-2" role="status"></div>
        ${t(textKey)}
    </div>`;
}

// 批量更新页面中的所有翻译
function updateAllTranslations() {
    waitForI18n().then(() => {
        // 更新所有带有 data-i18n 属性的元素
        $('[data-i18n]').each(function() {
            const key = $(this).attr('data-i18n');
            $(this).text(t(key));
        });
    });
}

// 更新所有国际化内容的函数（包括属性和placeholder）
function updateAllI18nContent() {
    // 更新所有带有 data-i18n 属性的元素
    $('[data-i18n]').each(function() {
        const key = $(this).attr('data-i18n');
        $(this).text(t(key));
    });
    
    // 更新所有带有 data-i18n-placeholder 属性的元素
    $('[data-i18n-placeholder]').each(function() {
        const key = $(this).attr('data-i18n-placeholder');
        $(this).attr('placeholder', t(key));
    });
    
    // 更新所有带有 data-i18n-title 属性的元素
    $('[data-i18n-title]').each(function() {
        const key = $(this).attr('data-i18n-title');
        $(this).attr('title', t(key));
    });
}

// 导出函数供全局使用
window.i18nHelper = {
    t,
    waitForI18n,
    createTranslatedOption,
    createTranslatedHTML,
    updateElementText,
    showTranslatedAlert,
    showTranslatedError,
    showTranslatedSuccess,
    createLoadingHTML,
    createLoadingMessage,
    updateAllTranslations,
    updateAllI18nContent
}; 