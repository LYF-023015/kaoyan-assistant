/**
 * 主应用逻辑
 */

const App = {
    currentConversationId: null,
    conversations: {},
    history: [],
    isGenerating: false,
    indexProgress: [],

    init() {
        this.loadConversations();
        this.bindEvents();
        this.renderHistoryList();
        this.loadFileList();
        ChatRenderer.init();
        this.updateHeaderModel();

        if (typeof renderMathInElement !== 'undefined') {
            window.renderMathInElement = renderMathInElement;
        }
    },

    updateHeaderModel() {
        const headerModel = document.getElementById('headerModel');
        if (headerModel) {
            headerModel.textContent = 'Kimi';
        }
    },

    bindEvents() {
        const sendBtn = document.getElementById('sendBtn');
        const messageInput = document.getElementById('messageInput');

        sendBtn.addEventListener('click', () => this.sendMessage());
        messageInput.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                this.sendMessage();
            }
        });
        messageInput.addEventListener('input', () => {
            sendBtn.disabled = messageInput.value.trim().length === 0;
            messageInput.style.height = 'auto';
            messageInput.style.height = Math.min(messageInput.scrollHeight, 200) + 'px';
        });

        document.getElementById('newChatBtn').addEventListener('click', () => {
            this.startNewConversation();
        });

        document.getElementById('themeToggle').addEventListener('click', () => {
            this.toggleTheme();
        });

        document.getElementById('uploadBtn').addEventListener('click', () => {
            document.getElementById('fileInput').click();
        });
        document.getElementById('fileInput').addEventListener('change', (e) => {
            this.handleFileUpload(e.target.files);
        });

        document.getElementById('indexBtn').addEventListener('click', () => {
            this.buildIndex();
        });

        document.getElementById('refreshFilesBtn').addEventListener('click', () => {
            this.loadFileList();
        });

        document.getElementById('subjectSelect').addEventListener('change', () => {
            this.loadFileList();
        });

        document.querySelectorAll('.welcome-card').forEach(card => {
            card.addEventListener('click', () => {
                const prompt = card.dataset.prompt;
                messageInput.value = prompt;
                sendBtn.disabled = false;
                this.sendMessage();
            });
        });
    },

    startNewConversation() {
        this.currentConversationId = Date.now().toString();
        this.conversations[this.currentConversationId] = {
            id: this.currentConversationId,
            title: '新对话',
            messages: [],
            subject: document.getElementById('subjectSelect').value,
            createdAt: new Date().toISOString(),
        };
        this.history = [];
        ChatRenderer.clearMessages();
        this.renderHistoryList();
        this.saveConversations();
    },

    async sendMessage() {
        const input = document.getElementById('messageInput');
        const content = input.value.trim();
        if (!content || this.isGenerating) return;

        if (!this.currentConversationId) {
            this.startNewConversation();
        }

        const conv = this.conversations[this.currentConversationId];

        if (conv.messages.length === 0) {
            conv.title = content.slice(0, 20) + (content.length > 20 ? '...' : '');
        }

        ChatRenderer.addUserMessage(content);
        conv.messages.push({ role: 'user', content });
        this.history.push({ role: 'user', content });

        input.value = '';
        input.style.height = 'auto';
        document.getElementById('sendBtn').disabled = true;
        this.isGenerating = true;

        const assistantMsg = ChatRenderer.addAssistantPlaceholder();

        const subject = document.getElementById('subjectSelect').value;
        const useWebSearch = document.getElementById('webSearchToggle').checked;

        let fullAnswer = '';

        try {
            await api.chatStream(
                content,
                this.history.slice(0, -1),
                subject,
                useWebSearch,
                (token) => {
                    fullAnswer += token;
                    ChatRenderer.updateAssistantMessage(assistantMsg, fullAnswer);
                },
                (answer, sources) => {
                    ChatRenderer.updateAssistantMessage(assistantMsg, answer, sources);
                    conv.messages.push({ role: 'assistant', content: answer });
                    this.history.push({ role: 'assistant', content: answer });
                    this.saveConversations();
                    this.renderHistoryList();
                    this.isGenerating = false;
                }
            );
        } catch (error) {
            console.error(error);
            ChatRenderer.updateAssistantMessage(
                assistantMsg,
                '抱歉，发生了错误：' + error.message + '\n\n请检查后端服务是否正常运行。'
            );
            this.isGenerating = false;
        }
    },

    async handleFileUpload(files) {
        if (!files || files.length === 0) return;
        const subject = document.getElementById('subjectSelect').value;
        let uploadCount = 0;

        for (const file of files) {
            this.showToast(`正在上传 ${file.name}...`);
            try {
                const result = await api.uploadFile(file, subject);
                if (result.success) {
                    this.showToast(`上传成功: ${file.name}`, 'success');
                    uploadCount++;
                } else {
                    this.showToast(`上传失败: ${result.message}`, 'error');
                }
            } catch (e) {
                this.showToast(`上传失败: ${e.message}`, 'error');
            }
        }
        document.getElementById('fileInput').value = '';
        this.loadFileList();

        // 上传完成后自动触发增量索引（只处理新文件）
        if (uploadCount > 0) {
            this.showToast(`正在自动索引 ${uploadCount} 个新文件...`);
            await this.buildIndex();
        }
    },

    async buildIndex() {
        const subject = document.getElementById('subjectSelect').value;
        const sysMsg = ChatRenderer.addSystemMessage('⏳ 正在开始构建索引...');
        const contentDiv = sysMsg.querySelector('.message-content');
        this.indexProgress = [];

        try {
            await api.buildIndexStream(
                subject,
                (data) => {
                    // 实时进度更新：只保留最近 8 条进度，避免消息过长
                    this.indexProgress.push(data);
                    if (this.indexProgress.length > 8) {
                        this.indexProgress.shift();
                    }

                    let html = '⏳ <strong>正在构建索引...</strong>';
                    html += '<div style="margin-top: 8px; max-height: 300px; overflow-y: auto;">';
                    for (const p of this.indexProgress) {
                        const isDone = p.status === 'done' || p.status === 'file_done';
                        const isError = p.status === 'error';
                        const color = isError ? '#ef4444' : (isDone ? '#22c55e' : 'var(--text-tertiary)');
                        html += `<div style="margin: 3px 0; padding: 4px 8px; background: var(--bg-primary); border-radius: 4px; font-size: 12px; color: ${color};">`;
                        html += this.escapeHtml(p.message);
                        if (p.current && p.total) {
                            const pct = Math.round((p.current / p.total) * 100);
                            html += `<div style="margin-top: 3px; height: 3px; background: var(--bg-hover); border-radius: 2px;"><div style="width: ${pct}%; height: 100%; background: var(--accent-color); border-radius: 2px; transition: width 0.3s;"></div></div>`;
                        }
                        html += '</div>';
                    }
                    html += '</div>';
                    contentDiv.innerHTML = html;
                },
                (result) => {
                    // 完成
                    if (result.success) {
                        let detailHtml = '';
                        if (result.details && result.details.length > 0) {
                            detailHtml = result.details.map(d => {
                                const chunkCount = d.total_chunks !== undefined ? d.total_chunks : (d.new_chunks || 0);
                                const fileList = d.files && d.files.length > 0
                                    ? d.files.slice(0, 5).join(', ') + (d.files.length > 5 ? ` 等共${d.files.length}个` : '')
                                    : '无';
                                const newLabel = d.new_chunks > 0 ? ` (新增 ${d.new_chunks})` : '';
                                return `<div style="margin: 4px 0; padding: 6px 10px; background: var(--bg-primary); border-radius: 6px; text-align: left;">
                                    <strong>[${d.subject}]</strong> ${chunkCount} 个文本块${newLabel}<br>
                                    <span style="font-size: 12px; color: var(--text-tertiary);">${fileList}</span>
                                </div>`;
                            }).join('');
                        }
                        contentDiv.innerHTML = `✅ <strong>索引构建完成</strong><div style="margin-top: 8px;">${this.escapeHtml(result.message)}</div>${detailHtml}`;
                        sysMsg.querySelector('.message-avatar').textContent = '✅';
                        this.loadFileList();
                    } else {
                        contentDiv.innerHTML = `❌ <strong>索引构建失败</strong><br>${this.escapeHtml(result.message)}`;
                        sysMsg.querySelector('.message-avatar').textContent = '❌';
                    }
                }
            );
        } catch (e) {
            contentDiv.innerHTML = `❌ <strong>索引构建失败</strong><br>${this.escapeHtml(e.message)}`;
            sysMsg.querySelector('.message-avatar').textContent = '❌';
        }
    },

    async loadFileList() {
        const subject = document.getElementById('subjectSelect').value;
        const fileListEl = document.getElementById('fileList');
        fileListEl.innerHTML = '<div style="padding: 8px; color: var(--text-tertiary); font-size: 12px;">加载中...</div>';

        try {
            const result = await api.getFiles(subject === '全部' ? null : subject);
            if (result.files && result.files.length > 0) {
                fileListEl.innerHTML = result.files.map(f => `
                    <div class="file-item" title="${this.escapeHtml(f.path)}">
                        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                            <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/>
                            <polyline points="14 2 14 8 20 8"/>
                        </svg>
                        <span>${this.escapeHtml(f.name)}</span>
                        <span class="file-size">${this.formatSize(f.size)}</span>
                    </div>
                `).join('');
            } else {
                fileListEl.innerHTML = '<div style="padding: 8px; color: var(--text-tertiary); font-size: 12px;">暂无文件</div>';
            }
        } catch (e) {
            fileListEl.innerHTML = '<div style="padding: 8px; color: var(--text-tertiary); font-size: 12px;">加载失败</div>';
        }
    },

    formatSize(bytes) {
        if (bytes < 1024) return bytes + ' B';
        if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB';
        return (bytes / (1024 * 1024)).toFixed(1) + ' MB';
    },

    toggleTheme() {
        const body = document.body;
        if (body.classList.contains('dark-theme')) {
            body.classList.remove('dark-theme');
            body.classList.add('light-theme');
            localStorage.setItem('theme', 'light');
        } else {
            body.classList.remove('light-theme');
            body.classList.add('dark-theme');
            localStorage.setItem('theme', 'dark');
        }
    },

    loadTheme() {
        const theme = localStorage.getItem('theme') || 'dark';
        document.body.classList.add(theme + '-theme');
    },

    renderHistoryList() {
        const list = document.getElementById('historyList');
        list.innerHTML = '';

        const sorted = Object.values(this.conversations).sort(
            (a, b) => new Date(b.createdAt) - new Date(a.createdAt)
        );

        for (const conv of sorted) {
            const li = document.createElement('li');
            li.className = 'history-item' + (conv.id === this.currentConversationId ? ' active' : '');
            li.innerHTML = `
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                    <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/>
                </svg>
                <span>${this.escapeHtml(conv.title)}</span>
            `;
            li.addEventListener('click', () => {
                this.loadConversation(conv.id);
            });
            list.appendChild(li);
        }
    },

    loadConversation(id) {
        const conv = this.conversations[id];
        if (!conv) return;

        this.currentConversationId = id;
        this.history = conv.messages.map(m => ({ role: m.role, content: m.content }));

        ChatRenderer.clearMessages();
        for (const msg of conv.messages) {
            if (msg.role === 'user') {
                ChatRenderer.addUserMessage(msg.content);
            } else {
                const placeholder = ChatRenderer.addAssistantPlaceholder();
                ChatRenderer.updateAssistantMessage(placeholder, msg.content);
            }
        }

        if (conv.subject) {
            document.getElementById('subjectSelect').value = conv.subject;
        }

        this.renderHistoryList();
    },

    saveConversations() {
        localStorage.setItem('kaoyan_conversations', JSON.stringify(this.conversations));
    },

    loadConversations() {
        try {
            const data = localStorage.getItem('kaoyan_conversations');
            if (data) {
                this.conversations = JSON.parse(data);
            }
        } catch (e) {
            this.conversations = {};
        }
    },

    showToast(message, type = 'info') {
        const toast = document.getElementById('toast');
        toast.textContent = message;
        toast.className = `toast ${type} show`;
        setTimeout(() => {
            toast.classList.remove('show');
        }, 3000);
    },

    escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    },
};

// 初始化
document.addEventListener('DOMContentLoaded', () => {
    App.loadTheme();
    App.init();
});
