$(document).ready(function() {
    moment.locale('zh-cn'); // Set moment locale

    // Translation helper function
    function t(key, fallback, params) {
        let text = window.i18n ? window.i18n.t(key, fallback) : (fallback || key);
        if (params) {
            Object.keys(params).forEach(param => {
                text = text.replace(new RegExp(`\\{${param}\\}`, 'g'), params[param]);
            });
        }
        return text;
    }

    let currentSelectedTaskId = null;
    let eventSource = null;
    let taskListRefreshInterval = null; // 添加任务列表刷新定时器变量
    const TASK_LIST_REFRESH_INTERVAL = 10000; // 每10秒刷新一次任务列表状态

    // --- Updated API Endpoints ---
    const API_ENDPOINTS = {
        // Use the corrected endpoints provided by the user
        sqlDatabasesAndTables: '/api/tables', // MODIFIED: Correct endpoint for tables
        knowledgeGraphs: '/api/kg/graphs',      // Endpoint to get KGs
        agentTasks: '/api/agent/tasks',
        agentTaskById: (id) => `/api/agent/tasks/${id}`,
        agentTaskMessages: (id) => `/api/agent/tasks/${id}/messages`,
        runAgentTask: (id) => `/api/agent/tasks/${id}/run`,
    };

    // --- Helper Functions (keep as before) ---
    function showToast(message, type = 'success') {
        // Replace with a more sophisticated toast library if available (e.g., Toastr, Bootstrap Toasts)
        // Simple alert for now:
        console.log(`Toast (${type}): ${message}`);
        // You might want to add a visual element here later
        const toastContainer = document.getElementById('toast-container'); // Assume you add a div with this ID in layout.html
        if (toastContainer) {
            const toastId = `toast-${Date.now()}`;
            const toastTypeClass = type === 'error' ? 'bg-danger' : (type === 'warning' ? 'bg-warning' : 'bg-success');
            const toastHtml = `
                <div id="${toastId}" class="toast align-items-center text-white ${toastTypeClass} border-0" role="alert" aria-live="assertive" aria-atomic="true">
                    <div class="d-flex">
                        <div class="toast-body">
                            ${escapeHtml(message)}
                        </div>
                        <button type="button" class="btn-close btn-close-white me-2 m-auto" data-bs-dismiss="toast" aria-label="Close"></button>
                    </div>
                </div>`;
            toastContainer.insertAdjacentHTML('beforeend', toastHtml);
            const toastElement = document.getElementById(toastId);
            const toast = new bootstrap.Toast(toastElement, { delay: 5000 });
            toast.show();
            toastElement.addEventListener('hidden.bs.toast', () => toastElement.remove());
        } else {
            alert(`${type.toUpperCase()}: ${message}`); // Fallback
        }
    }
    function formatDate(dateString) {
        return dateString ? moment(dateString).fromNow() : 'N/A';
    }
    function escapeHtml(unsafe) {
        if (unsafe === null || typeof unsafe === 'undefined') return '';
        return unsafe.toString().replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;").replace(/'/g, "&#039;");
    }
    function renderCode(code, language) {
        const safeCode = escapeHtml(code);
        // Ensure the language is valid for prism, fallback to 'clike' or 'text'
         const validLang = language && Prism.languages[language] ? language : 'clike';
        return `<pre><code class="language-${validLang}">${safeCode}</code></pre>`;
    }
    function getStatusBadgeClass(status) {
        switch (status) {
            case 'created': return 'bg-secondary';
            case 'running': return 'bg-primary';
            case 'completed': return 'bg-success';
            case 'failed': return 'bg-danger';
            default: return 'bg-light text-dark';
        }
    }

    // --- UI Update Functions ---
    function showLoading(elementId) { $(`#${elementId}`).show(); }
    function hideLoading(elementId) { $(`#${elementId}`).hide(); }

    // --- Updated Select Population Logic ---
    function populateDbTableSelect(data) {
        const $select = $('#sql-table-select'); // Use the new ID
        $select.empty().append(`<option value="">${t('data_decision.select_database_table', '-- 选择数据库表 --')}</option>`);
        // MODIFIED: Handle the flat list format {data: [{name: 't1'}, {name: 't2'}, ...]} from /api/tables
        if (data && data.data && Array.isArray(data.data)) {
             if (data.data.length > 0) {
                 // Get the configured database name if possible (e.g., from another API or assume single DB)
                 // For simplicity, we'll assume a single configured DB or omit it in the display value for now.
                 // Ideally, the backend /api/tables could also return the database name.
                 const dbName = "default"; // Placeholder or fetch actual DB name if needed elsewhere

                 data.data.forEach(tableInfo => {
                     const tableName = tableInfo.name;
                     if (tableName) {
                         // The value needs to contain DB name and table name for the backend agent creation
                         // We need the DB name. Let's assume it's available or make it a prerequisite.
                         // Hardcoding 'default' might not work if backend needs the actual configured DB name.
                         // A better approach is to fetch DB info separately or have /api/tables include it.
                         // TEMPORARY FIX: We need the DB name. Let's try fetching it first or require it.
                         // For now, we'll just use table name as value and handle parsing in create task.
                         // Or, better, let's assume the backend task creation can deduce the db from the table.
                         // Let's pass just the table name for now, and adjust backend if needed.
                         const value = tableName; // Simplification: Pass only table name
                         // OR const value = `${dbName}.${tableName}`; // If dbName is known

                         $select.append($('<option>', {
                             value: value, // Value used in form submission
                             text: escapeHtml(tableName) // Text displayed to user
                         }));
                     }
                 });
             } else {
                 $select.append(`<option value="" disabled>${t('data_decision.js.no_tables_in_database', '数据库中没有表')}</option>`);
             }
        } else {
             console.warn("Unrecognized format for Table data or data is empty:", data);
             $select.append(`<option value="" disabled>${t('data_decision.js.cannot_parse_table_list', '无法解析数据表列表')}</option>`);
        }
    }

    function populateKgSelect(graphs) { // Function expects the graphs array
        const $select = $('#kg-graph-select'); // Use the new ID
        $select.empty().append(`<option value="">${t('data_decision.select_knowledge_graph', '-- 选择知识图谱 --')}</option>`);
         // Assuming data is like [{id: 1, name: "Graph A", identifier: "1:Graph A"}, {id: 2, name: "Graph B", identifier: "2:Graph B"}]
         // We need the 'identifier' which the backend uses (e.g., "graph_id:graph_name")
        // MODIFICATION START: Iterate over the passed graphs array directly
        if (graphs && graphs.length > 0) {
            let optionsAdded = 0;
            graphs.forEach(graph => {
                // Check necessary fields
                const graphId = graph.id;
                const graphName = graph.name;
                const value = graph.identifier || (graphId !== undefined && graphName !== undefined ? `${graphId}:${graphName}` : undefined);

                if (value !== undefined && graphName !== undefined) {
                    $select.append($('<option>', {
                        value: value, // The identifier string
                        text: escapeHtml(graphName) + ` (ID: ${graphId || 'N/A'})`
                    }));
                    optionsAdded++;
                } else {
                    console.warn("Skipping invalid graph item for select (missing id, name, or identifier):", graph);
                }
            });
            // Check if any valid options were actually added
            if (optionsAdded === 0) {
                console.warn("No valid graph options could be added from the fetched data.");
                $select.append(`<option value="" disabled>${t('data_decision.js.invalid_kg_data', '获取到的图谱数据无效')}</option>`);
            }
        } else {
             // If the graphs array is empty
             $select.append(`<option value="" disabled>${t('data_decision.js.no_available_kg', '无可用知识图谱')}</option>`);
        }
        // MODIFICATION END
    }

    // --- renderTaskList, renderTaskStep, updateTaskStatusDisplay, clearTaskDetails (mostly unchanged, check details update) ---
     function renderTaskList(tasks) { // Keep as before
        const $container = $('#task-list-container');
        $container.empty();
        if (!tasks || tasks.length === 0) {
            $container.html(`<div class="list-group-item text-muted text-center">${t('data_decision.js.no_tasks_found', '没有找到任何任务。')}</div>`);
            return;
        }
        tasks.sort((a, b) => (b.session_id || 0) - (a.session_id || 0)); // Sort by ID descending

        tasks.forEach(task => {
            const statusClass = getStatusBadgeClass(task.status);
            const taskName = task.name || t('data_decision.js.unnamed_task', '未命名任务');
            const goalDescription = task.user_goal || t('data_decision.js.no_goal_description', '无目标描述');
            const deleteTaskTitle = t('data_decision.js.delete_task', '删除任务');
            
            const itemHtml = `
                <a href="#" class="list-group-item list-group-item-action task-list-item ${currentSelectedTaskId === task.session_id ? 'active' : ''}" data-task-id="${task.session_id}">
                    <div class="d-flex w-100 justify-content-between">
                        <h6 class="mb-1 text-truncate">${escapeHtml(taskName)} (ID: ${task.session_id})</h6>
                        <small title="${task.updated_at || task.created_at}">${formatDate(task.updated_at || task.created_at)}</small>
                    </div>
                    <p class="mb-1 text-truncate small">${escapeHtml(goalDescription)}</p>
                    <div class="d-flex justify-content-between align-items-center">
                         <small><span class="badge ${statusClass}">${escapeHtml(task.status || t('data_decision.js.unknown', '未知'))}</span></small>
                        <button class="btn btn-sm btn-outline-danger delete-task-btn" data-task-id="${task.session_id}" title="${deleteTaskTitle}">
                            <i class="fas fa-trash-alt"></i>
                        </button>
                    </div>
                </a>`;
            $container.append(itemHtml);
        });
        $('.task-list-item').on('click', handleTaskSelect);
        $('.delete-task-btn').on('click', handleDeleteTask);
    }
    function renderTaskStep(step) {
        const $container = $('#task-steps-container');
        let iconClass = 'fas fa-info-circle', title = t('data_decision.step.information', '信息'), contentHtml = '', cardClass = 'border-light';
        const stepType = step.message_type || step.type; // Allow 'type' from SSE data
        const rawContent = step.content || ''; // Get raw content

        // Use marked.parse for content types that should support Markdown
        const renderMarkdown = (content) => {
            if (typeof marked === 'undefined') {
                 console.error("marked.js is not loaded!");
                 // Fallback to basic escaping if marked is not available
                 return escapeHtml(content).replace(/\\n/g, '<br>');
            }
            marked.setOptions({ gfm: true, breaks: true });
            try {
                return marked.parse(content); // Parse Markdown
            } catch (e) {
                console.error("Markdown parsing error:", e);
                return `<p class="text-danger">${t('data_decision.step.markdown_parse_error', 'Markdown 解析错误:')}</p><pre>${escapeHtml(content)}</pre>`;
            }
        };

        switch (stepType) {
            case 'goal':
                iconClass = 'fas fa-bullseye'; title = t('data_decision.step.user_goal', '用户目标'); cardClass = 'border-secondary';
                contentHtml = `<p class="mb-0">${escapeHtml(rawContent)}</p>`; // Goal likely doesn't need Markdown
                break;
            case 'thought':
                 iconClass = 'fas fa-brain'; title = t('data_decision.step.thinking_process', '思考过程'); cardClass = 'border-info';
                contentHtml = renderMarkdown(rawContent); // Render thought as Markdown
                break;
            case 'action':
                 iconClass = 'fas fa-cogs'; title = `${t('data_decision.step.execute_action', '执行动作')}: ${escapeHtml(step.tool_name || step.tool || 'N/A')}`; cardClass = 'border-warning';
                // Action input is usually JSON, keep using renderCode
                contentHtml = `<strong>${t('data_decision.step.action_input', '工具输入:')}</strong> ${renderCode(JSON.stringify(step.tool_input, null, 2), 'json')}`;
                break;
            case 'observation':
                iconClass = 'fas fa-binoculars'; title = `${t('data_decision.step.observation_result', '观察结果')} (${escapeHtml(step.tool_name || step.tool || 'N/A')})`; cardClass = 'border-primary';
                 // Render observation as Markdown, this will handle tables, code blocks etc.
                 // The previous code block regex replacements are no longer needed here.
                 contentHtml = renderMarkdown(rawContent);
                break;
            case 'final_answer':
                iconClass = 'fas fa-check-circle'; title = t('data_decision.step.final_answer', '最终答案'); cardClass = 'border-success';
                 contentHtml = renderMarkdown(rawContent); // Render final answer as Markdown
                break;
            case 'error':
                iconClass = 'fas fa-exclamation-triangle'; title = t('data_decision.step.error', '错误'); cardClass = 'border-danger'; // Remove text-danger class from card, apply to content if needed
                // Render error content as Markdown
                contentHtml = `<div class="text-danger">${renderMarkdown(rawContent)}</div>`;
                if (step.tool_name === 'interrupt' || step.tool_name === 'system_stop') {
                    title = t('data_decision.step.task_interrupted', '任务已中断'); iconClass = 'fas fa-stop-circle';
                } else if (step.tool_name === 'system_cleanup') {
                    title = t('data_decision.step.system_cleanup', '系统清理/失败回退'); iconClass = 'fas fa-broom';
                }
                if(step.llm_output) {
                    contentHtml += `<br><small>LLM Output:</small>${renderCode(step.llm_output, 'text')}`;
                }
                break;
            case 'warning':
                iconClass = 'fas fa-exclamation-circle'; title = t('data_decision.step.warning', '警告'); cardClass = 'border-warning'; // Remove text-warning class
                // Render warning content as Markdown
                contentHtml = `<div class="text-warning">${renderMarkdown(rawContent)}</div>`;
                break;
            case 'system':
                 iconClass = 'fas fa-info-circle'; title = t('data_decision.step.system_message', '系统消息'); cardClass = 'border-secondary';
                 // Render system message as Markdown
                 contentHtml = `<div class="fst-italic">${renderMarkdown(rawContent)}</div>`;
                break;
            case 'status': updateTaskStatusDisplay(step.content); return; // Don't render status as card
            default:
                console.warn("Unknown step type:", stepType, step);
                title = `${t('data_decision.step.unknown_type', '未知')} (${escapeHtml(stepType)})`;
                // Display raw JSON for unknown types, escaped
                contentHtml = `<pre>${escapeHtml(JSON.stringify(step, null, 2))}</pre>`;
        }
        const timestamp = step.timestamp ? moment(step.timestamp).format('YYYY-MM-DD HH:mm:ss') : moment().format('HH:mm:ss');
        const stepHtml = `
            <div class="card mb-2 agent-step-card ${cardClass}">
                <div class="card-body p-2">
                     <div class="d-flex justify-content-between align-items-center mb-1">
                         <small class="text-muted"><i class="${iconClass} me-1"></i><strong>${title}</strong></small>
                         <small class="text-muted">${timestamp}</small>
                     </div>
                     <div class="agent-step-content">${contentHtml}</div>
                 </div>
            </div>`;
        $container.append(stepHtml);

        // IMPORTANT: Highlight code blocks AFTER marked.js has rendered the HTML
        // Find the newly added card and highlight code blocks within its content
        const newStepCard = $container.children().last()[0];
        if (newStepCard) {
             Prism.highlightAllUnder(newStepCard.querySelector('.agent-step-content'));
        }
    }
    function updateTaskStatusDisplay(status) {
         const $footer = $('#task-status-footer');
         const $statusSpan = $('#current-task-status');
         const $runButton = $('#run-task-btn');
         const $stopButton = $('#stop-task-btn'); // Get stop button
         const statusClass = getStatusBadgeClass(status);

         $statusSpan.text(status).removeClass().addClass(`badge ${statusClass}`);
         $footer.show();

         if (status === 'running') {
             $runButton.prop('disabled', true).html(`<span class="spinner-border spinner-border-sm me-1"></span>${t('data_decision.js.running_status', '运行中...')}`);
             $stopButton.prop('disabled', false).show(); // Enable and show stop button
         } else {
             $runButton.prop('disabled', !currentSelectedTaskId || status === 'running').html(`<i class="fas fa-play me-1"></i>${t('data_decision.run_task', '运行任务')}`);
             $stopButton.prop('disabled', true).hide(); // Disable and hide stop button
         }
         // Update info section status badge too
         $('#task-info-status').text(status).removeClass().addClass(`badge ${statusClass}`);
     }
    function clearTaskDetails() { // Keep as before
         $('#task-detail-title').text(t('data_decision.task_details', '任务详情'));
         $('#task-select-prompt').show();
         $('#task-info').hide();
         $('#task-steps-container').empty();
         $('#task-status-footer').hide();
         $('#run-task-btn').prop('disabled', true);
         $('#stop-task-btn').hide().prop('disabled', true); // Also hide/disable stop button
         currentSelectedTaskId = null;
         $('.task-list-item').removeClass('active');
         if (eventSource) { eventSource.close(); eventSource = null; console.log("Closed previous EventSource connection."); }
     }


    // --- API Call Functions (Updated Fetch Functions) ---
    async function fetchTables() { // Renamed and updated endpoint/logic
        try {
            const response = await fetch(API_ENDPOINTS.sqlDatabasesAndTables); // Use the correct endpoint '/api/tables'
            if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
            const data = await response.json(); // Expecting {status: 'success', data: [{name: 't1'}, ...]}
            populateDbTableSelect(data); // Pass the whole response object
        } catch (error) {
            console.error('Error fetching Tables:', error);
             // Update select ID in error message
            $('#sql-table-select').empty().append(`<option value="" disabled>${t('data_decision.js.load_db_tables_failed', '加载数据库/表失败')}</option>`);
            showToast(t('data_decision.js.cannot_load_db_tables', '无法加载数据库/表列表'), 'error');
        }
    }

    async function fetchKnowledgeGraphs() { // Renamed and updated endpoint
        try {
            const response = await fetch(API_ENDPOINTS.knowledgeGraphs); // Use new endpoint
            if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
            const data = await response.json(); // data should be {status: 'success', graphs: [...]

            // *** MODIFICATION START ***
            // Explicitly check for status and the graphs array
            if (data && data.status === 'success' && Array.isArray(data.graphs)) {
                 console.log("Fetched knowledge graphs:", data.graphs);
                 populateKgSelect(data.graphs); // Pass the graphs array
            } else {
                 // Log the actual received data for debugging
                 console.error('Error fetching Knowledge Graphs: Invalid data format received', data);
                 $('#kg-graph-select').empty().append(`<option value="" disabled>${t('data_decision.js.cannot_parse_table_list', '无法解析知识图谱列表')}</option>`);
                 showToast(t('data_decision.js.invalid_kg_data_format', '无法加载知识图谱列表: 返回格式错误'), 'error');
            }
            // *** MODIFICATION END ***
        } catch (error) {
            console.error('Error fetching Knowledge Graphs:', error);
             // Update select ID in error message
            $('#kg-graph-select').empty().append(`<option value="" disabled>${t('data_decision.js.load_kg_failed', '加载知识图谱失败')}</option>`);
            showToast(t('data_decision.js.cannot_load_kg_list', '无法加载知识图谱列表'), 'error');
        }
    }

    async function fetchAgentTasks() { // Keep as before
        showLoading('task-list-loading');
        try {
            const response = await fetch(API_ENDPOINTS.agentTasks);
            if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
            const data = await response.json();
            renderTaskList(data);
        } catch (error) {
            console.error('Error fetching agent tasks:', error);
            $('#task-list-container').html(`<div class="list-group-item text-danger text-center">${t('data_decision.js.load_task_list_failed', '加载任务列表失败。')}</div>`);
             showToast(t('data_decision.js.load_task_list_failed', '加载任务列表失败'), 'error');
        } finally {
             hideLoading('task-list-loading');
        }
    }

    // --- Updated Task Creation Data ---
    async function createTask(taskData) {
        try {
            const response = await fetch(API_ENDPOINTS.agentTasks, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(taskData)
            });
            const result = await response.json();
            if (!response.ok) {
                 throw new Error(result.error || `HTTP error! status: ${response.status}`);
            }
            showToast(t('data_decision.js.task_created_success', '任务创建成功!'), 'success');
            $('#create-task-form')[0].reset(); // Clear form
            // Reset max_iterations to default visually (since reset() might not trigger default value)
            $('#max-iterations').val(20);
            fetchAgentTasks(); // Refresh list
        } catch (error) {
             console.error('Error creating agent task:', error);
            showToast(`${t('data_decision.js.task_creation_failed', '创建任务失败')}: ${error.message}`, 'error');
        }
    }

     async function deleteTask(taskId) { // Keep as before
         if (!confirm(t('data_decision.js.confirm_delete_task', '确定要删除任务 ID: {taskId} 吗？', {taskId}))) return;
         try {
             const response = await fetch(API_ENDPOINTS.agentTaskById(taskId), { method: 'DELETE' });
             if (!response.ok) { // Check status directly
                  const result = await response.json().catch(() => ({})); // Try to parse error
                 throw new Error(result.error || `HTTP error! status: ${response.status}`);
             }
             showToast(t('data_decision.js.task_deleted', '任务 {taskId} 已删除', {taskId}), 'success');
             if (currentSelectedTaskId === taskId) clearTaskDetails();
             fetchAgentTasks();
         } catch (error) {
             console.error(`Error deleting task ${taskId}:`, error);
             showToast(`${t('data_decision.js.delete_task_failed', '删除任务失败')}: ${error.message}`, 'error');
         }
     }

     // --- Updated Task Detail Display ---
     async function fetchTaskDetailsAndMessages(taskId) {
         clearTaskDetails();
         currentSelectedTaskId = taskId;
         $('.task-list-item').removeClass('active');
         $(`.task-list-item[data-task-id="${taskId}"]`).addClass('active');
         $('#task-select-prompt').hide();
         $('#task-steps-container').html(`<div class="text-center text-muted p-3"><div class="spinner-border spinner-border-sm"></div> ${t('data_decision.loading', '加载中...')}</div>`);

         try {
             const metaResponse = await fetch(API_ENDPOINTS.agentTaskById(taskId));
             if (!metaResponse.ok) throw new Error(`HTTP error! status: ${metaResponse.status}`);
             const taskMeta = await metaResponse.json();

             const taskName = taskMeta.name || t('data_decision.js.unnamed', '未命名');
             $('#task-detail-title').text(t('data_decision.js.task_name_id', '任务: {name} (ID: {taskId})', {name: taskName, taskId}));
             $('#task-info-goal').text(taskMeta.user_goal || 'N/A');
             const sqlContext = taskMeta.sql_database_name && taskMeta.sql_table_name
                 ? `${escapeHtml(taskMeta.sql_database_name)}.${escapeHtml(taskMeta.sql_table_name)}`
                 : t('data_decision.js.none', '无');
             const kgContext = taskMeta.kg_graph_name ? escapeHtml(taskMeta.kg_graph_name) : t('data_decision.js.none', '无');
             $('#task-info-sql').text(sqlContext);
             $('#task-info-kg').text(kgContext);
             updateTaskStatusDisplay(taskMeta.status || 'unknown'); // This will handle button states
             $('#task-info').show();
             // Run button state is handled by updateTaskStatusDisplay
             // Stop button state is handled by updateTaskStatusDisplay

             const messagesResponse = await fetch(API_ENDPOINTS.agentTaskMessages(taskId));
             if (!messagesResponse.ok) throw new Error(`HTTP error! status: ${messagesResponse.status}`);
             const messages = await messagesResponse.json();
             const $stepsContainer = $('#task-steps-container');
             $stepsContainer.empty();
             if (messages && messages.length > 0) {
                 messages.sort((a, b) => (a.id || 0) - (b.id || 0)); // Sort messages by ID
                 messages.forEach(msg => renderTaskStep(msg));
             } else {
                 $stepsContainer.html(`<p class="text-muted text-center">${t('data_decision.js.no_execution_steps', '此任务还没有执行步骤。')}</p>`);
             }
         } catch (error) {
             console.error(`Error fetching details for task ${taskId}:`, error);
             showToast(`${t('data_decision.js.load_task_details_failed', '加载任务详情失败')}: ${error.message}`, 'error');
             $('#task-steps-container').html(`<p class="text-danger text-center">${t('data_decision.js.cannot_load_task_steps', '无法加载任务步骤。')}</p>`);
             $('#run-task-btn').prop('disabled', true);
             $('#stop-task-btn').hide().prop('disabled', true); // Ensure stop is hidden on error
             currentSelectedTaskId = null;
             $('.task-list-item').removeClass('active');
         }
     }

    // --- NEW API Call Function ---
    async function stopAgentTask(taskId) {
        console.log(`Requesting stop for task ${taskId}`);
        const $stopButton = $('#stop-task-btn');
        $stopButton.prop('disabled', true).html(`<span class="spinner-border spinner-border-sm me-1"></span>${t('data_decision.js.stopping_task', '正在终止...')}`);

        try {
            const response = await fetch(API_ENDPOINTS.agentTaskById(taskId) + '/stop', { // Corrected endpoint construction
                method: 'POST',
                headers: { 'Content-Type': 'application/json' }
                // No body needed for this request based on backend
            });
            const result = await response.json(); // Expect {message: ...} or {error: ...}
            if (!response.ok) {
                throw new Error(result.error || `HTTP error! status: ${response.status}`);
            }
            showToast(t('data_decision.js.task_stop_request_sent', '任务 {taskId} 终止请求已发送', {taskId}), 'success');
            // The status should update via SSE when the task actually stops and fails
            // Optionally, force a status check or rely on SSE
        } catch (error) {
            console.error(`Error stopping task ${taskId}:`, error);
            showToast(`${t('data_decision.js.stop_task_failed', '请求终止任务失败')}: ${error.message}`, 'error');
            // Re-enable stop button only if the task is still potentially running
            const currentStatus = $('#current-task-status').text();
            if (currentStatus === 'running') {
                 $stopButton.prop('disabled', false).html(`<i class="fas fa-stop me-1"></i>${t('data_decision.stop', '终止')}`);
            }
        }
    }


    // --- Event Handlers ---
    $('#create-task-form').on('submit', function(event) {
        event.preventDefault();
        const selectedTableValue = $('#sql-table-select').val();
        let dbName = null;
        let tableName = selectedTableValue; // Assume value is just table name
        // Backend needs to handle null dbName if needed

        const maxIterations = parseInt($('#max-iterations').val(), 10) || 20; // Get value, default 20

        const taskData = {
            name: $('#task-name').val() || null,
            user_goal: $('#user-goal').val(),
            sql_database_name: dbName,
            sql_table_name: tableName || null, // Send null if empty
            kg_graph_name: $('#kg-graph-select').val() || null,
            max_iterations: maxIterations // Add max_iterations
        };

        if (!taskData.user_goal) {
            showToast(t('data_decision.js.task_goal_empty', '任务目标不能为空'), 'warning');
            return;
        }
        createTask(taskData);
    });

     function handleTaskSelect(event) { // Keep as before
         event.preventDefault();
         const $target = $(event.target);
         const $item = $target.closest('.task-list-item');
         const taskId = parseInt($item.data('task-id'));
         if (taskId && taskId !== currentSelectedTaskId) {
              fetchTaskDetailsAndMessages(taskId);
         }
     }

     function handleDeleteTask(event) { // Keep as before
         event.preventDefault();
         event.stopPropagation();
         const taskId = parseInt($(this).data('task-id'));
         if (taskId) {
             deleteTask(taskId);
         }
     }

     $('#run-task-btn').on('click', function() {
         if (!currentSelectedTaskId) {
             showToast(t('data_decision.js.please_select_task', '请先选择一个任务'), 'warning');
             return;
         }
         console.log(`Running task ${currentSelectedTaskId}`);
         updateTaskStatusDisplay('running'); // This will show/enable stop button
         $('#task-steps-container').empty(); // Clear previous steps before running
         if (eventSource) eventSource.close();

         try {
             eventSource = new EventSource(API_ENDPOINTS.runAgentTask(currentSelectedTaskId));

             eventSource.onopen = () => {
                 console.log("SSE connection opened for task", currentSelectedTaskId);
                 $('#task-steps-container').append(`<p class="text-success text-center small">${t('data_decision.js.connection_established', '已连接，等待 Agent 输出...')}</p>`);
            };

             eventSource.onmessage = (event) => {
                 try {
                     const data = JSON.parse(event.data);
                     console.log("SSE data:", data);
                     // Remove initial "waiting" message if present
                     $('#task-steps-container p.text-success.small').remove();
                     renderTaskStep(data); // Render new step
                 } catch (e) {
                     console.error("SSE parse error:", e, event.data);
                     renderTaskStep({ type: 'error', content: `${t('data_decision.js.invalid_server_data', '从服务器接收到无效数据')}: ${event.data}` });
                 }
             };
             eventSource.addEventListener('error', (event) => {
                 console.error("SSE error:", event);
                 let errorMsg = t('data_decision.js.connection_error', '与服务器的连接发生错误。');
                 let finalStatus = 'failed'; // Assume failure on error
                 // Remove waiting message if present
                 $('#task-steps-container p.text-success.small').remove();

                 if (eventSource && eventSource.readyState === EventSource.CLOSED) {
                     errorMsg = t('data_decision.js.connection_closed', '连接已关闭。任务可能已完成、失败或被中断。');
                     // Don't assume 'failed' if closed, status should come from 'end' or prior 'status' message
                     finalStatus = $('#current-task-status').text() || t('data_decision.js.unknown', '未知'); // Use last known status
                     console.log("SSE closed, last known status:", finalStatus);
                 } else {
                     // Render error only if it's not a clean closure
                      renderTaskStep({ type: 'error', content: errorMsg });
                 }
                 // Update status display, but avoid overriding a 'completed' status received just before close
                 if (finalStatus !== 'completed') {
                    updateTaskStatusDisplay(finalStatus);
                 }
                 $('#run-task-btn').prop('disabled', finalStatus === 'running'); // Re-enable run if not running
                 $('#stop-task-btn').prop('disabled', true).hide(); // Disable/hide stop on error/close

                 if (eventSource) { eventSource.close(); eventSource = null; }
             });
              eventSource.addEventListener('end', (event) => { // Custom end event from backend
                 console.log("SSE stream ended by server signal.");
                  // Remove waiting message if present
                 $('#task-steps-container p.text-success.small').remove();

                 let finalStatusData = {};
                 try { finalStatusData = JSON.parse(event.data || '{}'); } catch(e) {}
                 const finalStatus = finalStatusData.status || $('#current-task-status').text() || t('data_decision.js.unknown', '未知'); // Get status from event data or last known

                 renderTaskStep({ type: 'system', content: t('data_decision.js.task_stream_ended', '任务执行流结束 (最终状态: {status})。', {status: finalStatus}) });
                 updateTaskStatusDisplay(finalStatus); // Ensure final status is reflected
                 $('#run-task-btn').prop('disabled', false).html(`<i class="fas fa-play me-1"></i>${t('data_decision.run_task', '运行任务')}`);
                 $('#stop-task-btn').prop('disabled', true).hide(); // Disable/hide stop button
                 if (eventSource) { eventSource.close(); eventSource = null; }
             });

         } catch (e) {
              console.error("Failed to create EventSource:", e);
              showToast(t('data_decision.js.cannot_start_task_stream', '无法启动任务执行流'), "error");
              updateTaskStatusDisplay('failed');
              $('#run-task-btn').prop('disabled', false).html(`<i class="fas fa-play me-1"></i>${t('data_decision.run_task', '运行任务')}`);
              $('#stop-task-btn').prop('disabled', true).hide();
         }
     });

    // --- NEW Event Handler for Stop Button ---
    $('#stop-task-btn').on('click', function() {
        if (currentSelectedTaskId) {
            stopAgentTask(currentSelectedTaskId);
        } else {
            showToast(t('data_decision.js.no_selected_task', '没有选中的任务可终止'), 'warning');
        }
    });


    // --- Initialization ---
    function initializePage() {
        console.log("Initializing Data Decision page...");
        fetchTables(); // MODIFIED: Call the renamed function
        fetchKnowledgeGraphs();    // Fetch KGs instead of KG sessions
        fetchAgentTasks();
        clearTaskDetails(); // Start with no task selected
         // Update HTML select element IDs
         $('#sql-session-select').attr('id', 'sql-table-select');
         $('#kg-session-select').attr('id', 'kg-graph-select');
         // Update labels if needed (optional)
          $('label[for="sql-session-select"]').attr('for', 'sql-table-select').text(t('data_decision.associated_database_table', '关联数据库表 (可选)'));
          $('label[for="kg-session-select"]').attr('for', 'kg-graph-select').text(t('data_decision.associated_knowledge_graph', '关联知识图谱 (可选)'));
          
        // 设置定时器自动刷新任务列表
        if (taskListRefreshInterval) {
            clearInterval(taskListRefreshInterval);
        }
        taskListRefreshInterval = setInterval(fetchAgentTasks, TASK_LIST_REFRESH_INTERVAL);
        console.log("Initialization complete.");
    }

    // 页面卸载时清除定时器
    $(window).on('unload', function() {
        if (taskListRefreshInterval) {
            clearInterval(taskListRefreshInterval);
            taskListRefreshInterval = null;
        }
        if (eventSource) {
            eventSource.close();
            eventSource = null;
        }
    });

    initializePage();

});
