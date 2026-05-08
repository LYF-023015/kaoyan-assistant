/**
 * 聊天消息渲染与处理
 */

// 配置 marked
marked.setOptions({
    breaks: true,
    gfm: true,
    headerIds: false,
});

const ChatRenderer = {
    messagesWrapper: document.getElementById('messagesWrapper'),
    chatContainer: document.getElementById('chatContainer'),
    welcomeScreen: document.getElementById('welcomeScreen'),

    init() {
        // 绑定代码复制
        this.messagesWrapper.addEventListener('click', (e) => {
            if (e.target.classList.contains('code-copy-btn')) {
                const code = e.target.closest('pre')?.querySelector('code')?.innerText;
                if (code) {
                    navigator.clipboard.writeText(code).then(() => {
                        e.target.textContent = '已复制';
                        setTimeout(() => e.target.textContent = '复制', 2000);
                    });
                }
            }
        });
    },

    hideWelcome() {
        if (this.welcomeScreen) {
            this.welcomeScreen.style.display = 'none';
        }
    },

    addUserMessage(content) {
        this.hideWelcome();
        const msgDiv = document.createElement('div');
        msgDiv.className = 'message user';
        msgDiv.innerHTML = `
            <div class="message-avatar">我</div>
            <div class="message-content">${this.escapeHtml(content)}</div>
        `;
        this.messagesWrapper.appendChild(msgDiv);
        this.scrollToBottom();
        return msgDiv;
    },

    addSystemMessage(content, type = 'info') {
        this.hideWelcome();
        const msgDiv = document.createElement('div');
        msgDiv.className = 'message system';
        const icon = type === 'success' ? '✅' : type === 'error' ? '❌' : 'ℹ️';
        msgDiv.innerHTML = `
            <div class="message-avatar" style="background: var(--bg-hover); color: var(--text-tertiary); border: 1px solid var(--border-color);">${icon}</div>
            <div class="message-content" style="background: var(--bg-hover); border: 1px solid var(--border-color); color: var(--text-secondary); font-size: 14px;">
                ${content}
            </div>
        `;
        this.messagesWrapper.appendChild(msgDiv);
        this.scrollToBottom();
        return msgDiv;
    },

    addAssistantPlaceholder() {
        this.hideWelcome();
        const msgDiv = document.createElement('div');
        msgDiv.className = 'message assistant';
        msgDiv.innerHTML = `
            <div class="message-avatar">
                <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                    <path d="M12 2L2 7l10 5 10-5-10-5z"/>
                    <path d="M2 17l10 5 10-5"/>
                    <path d="M2 12l10 5 10-5"/>
                </svg>
            </div>
            <div class="message-content">
                <div class="thinking-indicator">
                    思考中
                    <div class="dots">
                        <div class="dot"></div>
                        <div class="dot"></div>
                        <div class="dot"></div>
                    </div>
                </div>
            </div>
        `;
        this.messagesWrapper.appendChild(msgDiv);
        this.scrollToBottom();
        return msgDiv;
    },

    updateAssistantMessage(msgDiv, content, sources = null) {
        const contentDiv = msgDiv.querySelector('.message-content');
        const html = this.renderMarkdown(content);
        contentDiv.innerHTML = html;

        // 渲染 KaTeX
        if (window.renderMathInElement) {
            renderMathInElement(contentDiv, {
                delimiters: [
                    { left: '$$', right: '$$', display: true },
                    { left: '$', right: '$', display: false },
                    { left: '\\[', right: '\\]', display: true },
                    { left: '\\(', right: '\\)', display: false },
                ],
                throwOnError: false,
            });
        }

        // 高亮代码
        contentDiv.querySelectorAll('pre code').forEach((block) => {
            hljs.highlightElement(block);
        });

        // 添加引用来源
        if (sources && sources.length > 0) {
            this.addSources(contentDiv, sources);
        }

        this.scrollToBottom();
    },

    addSources(contentDiv, sources) {
        const sourcesId = 'sources-' + Date.now();
        const sourcesHtml = `
            <div class="sources-section">
                <div class="sources-toggle" onclick="document.getElementById('${sourcesId}').classList.toggle('open')">
                    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                        <polyline points="6 9 12 15 18 9"/>
                    </svg>
                    参考来源 (${sources.length})
                </div>
                <div class="sources-list" id="${sourcesId}">
                    ${sources.map(s => `
                        <div class="source-item">
                            <div class="source-meta">${s.metadata?.source || '未知'} · 相关度: ${(s.score || 0).toFixed(4)}</div>
                            <div class="source-text">${this.escapeHtml(s.content || '')}</div>
                        </div>
                    `).join('')}
                </div>
            </div>
        `;
        contentDiv.insertAdjacentHTML('beforeend', sourcesHtml);
    },

    renderMarkdown(text) {
        // 步骤1：保护 LaTeX 公式，防止 marked 破坏其中的 _ 等字符
        const formulas = [];
        let processed = text;

        // 保护 $$...$$
        processed = processed.replace(/\$\$([\s\S]*?)\$\$/g, (match, formula) => {
            formulas.push({ type: 'display', content: formula });
            return `\x00FORMULA_${formulas.length - 1}_DISPLAY\x00`;
        });

        // 保护 $...$（不在 $$ 内部）
        processed = processed.replace(/(?<!\$)\$(?!\$)(.*?)\$(?!\$)/g, (match, formula) => {
            formulas.push({ type: 'inline', content: formula });
            return `\x00FORMULA_${formulas.length - 1}_INLINE\x00`;
        });

        // 步骤2：预处理代码块，添加复制按钮
        processed = processed.replace(/```(\w+)?\n([\s\S]*?)```/g, (match, lang, code) => {
            const language = lang || 'text';
            return `<pre><div class="code-header"><span>${language}</span><button class="code-copy-btn">复制</button></div><code class="language-${language}">${this.escapeHtml(code.trim())}</code></pre>`;
        });

        // 步骤3：行内代码（排除已保护的公式区域）
        processed = processed.replace(/`([^`]+)`/g, '<code>$1</code>');

        // 步骤4：使用 marked 解析 Markdown
        let html = marked.parse(processed);

        // 步骤5：恢复公式占位符为 KaTeX 可识别的格式
        formulas.forEach((f, i) => {
            const placeholder = f.type === 'display'
                ? `\x00FORMULA_${i}_DISPLAY\x00`
                : `\x00FORMULA_${i}_INLINE\x00`;
            const replacement = f.type === 'display'
                ? `$$${f.content}$$`
                : `$${f.content}$`;
            html = html.replace(placeholder, replacement);
        });

        return html;
    },

    escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    },

    scrollToBottom() {
        this.chatContainer.scrollTop = this.chatContainer.scrollHeight;
    },

    clearMessages() {
        this.messagesWrapper.innerHTML = '';
        if (this.welcomeScreen) {
            this.welcomeScreen.style.display = 'flex';
        }
    },
};
