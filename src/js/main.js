// 导入依赖的库
import $ from 'jquery';
window.$ = window.jQuery = $; // 使jQuery全局可用

// 导入Bootstrap（需要jQuery和Popper.js）
import 'bootstrap';

// 导入样式
// import 'bootstrap/dist/css/bootstrap.min.css'; // Bootstrap CSS通过CopyPlugin复制，此处无需导入
import '@fortawesome/fontawesome-free/css/all.min.css'; // 导入Font Awesome CSS

// 导出通用函数或模块
export function initSidebar() {
    // 侧边栏折叠/展开功能
    $('#sidebar-toggle').on('click', function() {
        $('body').toggleClass('body-sidebar-collapsed');
        
        // 保存状态到本地存储
        try {
            localStorage.setItem('sidebarCollapsed', $('body').hasClass('body-sidebar-collapsed'));
        } catch (e) {
            console.error('Error saving sidebar state to localStorage:', e);
        }
    });
    
    // 页面加载时应用之前保存的状态
    try {
        if (localStorage.getItem('sidebarCollapsed') === 'true') {
            $('body').addClass('body-sidebar-collapsed');
        }
    } catch (e) {
        console.error('Error reading sidebar state from localStorage:', e);
    }
}

// 新增：设置侧边栏活动链接的函数
function setActiveSidebarLink() {
    const currentPath = window.location.pathname;
    // console.log("Current Path:", currentPath); // Debugging

    // 移除所有链接的 active 类
    $('.sidebar .nav-link').removeClass('active');

    // 遍历所有链接
    $('.sidebar .nav-link').each(function() {
        const linkHref = $(this).attr('href');
        // console.log("Checking Link Href:", linkHref); // Debugging
        
        // 检查链接的路径是否与当前路径完全匹配
        // 特殊处理首页：如果当前路径是 /，则首页链接也匹配
        if (linkHref === currentPath || (currentPath === '/' && linkHref === '/')) {
             // console.log("Exact match found:", linkHref); // Debugging
            $(this).addClass('active');
            return false; // 找到匹配项，停止遍历
        }
        
        // 尝试路径前缀匹配 (排除根路径 `/` 的简单前缀匹配，避免所有链接都激活)
        // 确保 linkHref 和 currentPath 都不是简单的 `/`
        if (linkHref !== '/' && currentPath !== '/' && currentPath.startsWith(linkHref)) {
             // console.log("Prefix match found for:", linkHref, "in", currentPath); // Debugging
             // 考虑是否需要前缀匹配，如果需要，取消下面的注释
             // $(this).addClass('active');
             // return false; 
        }
    });
    
    // 如果没有任何链接匹配 (比如在根路径 `/` 但没有完全匹配的 `/` 链接时)
    // 再次检查首页链接
    if ($('.sidebar .nav-link.active').length === 0 && currentPath === '/') {
        $('.sidebar .nav-link[href="/"]').addClass('active');
    }
}

// 自动初始化所有通用功能
$(document).ready(function() {
    initSidebar();
    setActiveSidebarLink(); // 在页面加载时调用设置活动链接的函数
    
    // 如果页面包含带有data-bs-toggle属性的元素，初始化Bootstrap组件
    $('[data-bs-toggle="tooltip"]').tooltip();
    $('[data-bs-toggle="popover"]').popover();
    
    console.log('WisdominDATA 前端脚本已加载');
}); 