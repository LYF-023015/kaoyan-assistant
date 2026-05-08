/**
 * API 调用封装
 */
const API_BASE = window.location.origin + '/api';

const api = {
    async chatStream(message, history, subject, useWebSearch, onToken, onDone) {
        const response = await fetch(`${API_BASE}/chat/stream`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                message,
                history,
                subject,
                use_web_search: useWebSearch,
            }),
        });

        if (!response.ok) {
            throw new Error(`HTTP ${response.status}`);
        }

        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let buffer = '';

        while (true) {
            const { done, value } = await reader.read();
            if (done) break;

            buffer += decoder.decode(value, { stream: true });
            const lines = buffer.split('\n\n');
            buffer = lines.pop() || '';

            for (const line of lines) {
                if (line.startsWith('data: ')) {
                    const dataStr = line.slice(6);
                    try {
                        const data = JSON.parse(dataStr);
                        if (data.done) {
                            onDone && onDone(data.answer, data.sources);
                        } else if (data.token !== undefined) {
                            onToken && onToken(data.token);
                        } else if (data.error) {
                            onToken && onToken(`\n[错误: ${data.error}]`);
                            onDone && onDone(`[错误: ${data.error}]`, []);
                            return;
                        }
                    } catch (e) {
                        // 忽略解析失败的行
                    }
                }
            }
        }
    },

    async uploadFile(file, subject) {
        const formData = new FormData();
        formData.append('file', file);
        formData.append('subject', subject);

        const response = await fetch(`${API_BASE}/upload`, {
            method: 'POST',
            body: formData,
        });
        return response.json();
    },

    // SSE 流式索引构建
    buildIndexStream(subject, onProgress, onDone) {
        const eventSource = new EventSource(`${API_BASE}/index?subject=${encodeURIComponent(subject)}`, {
            // 注意：EventSource 不支持 POST，需要改接口为 GET 或用 fetch 手动解析 SSE
        });

        // 由于 EventSource 不支持 POST，我们改用 fetch 手动解析 SSE
        return fetch(`${API_BASE}/index`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ subject }),
        }).then(response => {
            const reader = response.body.getReader();
            const decoder = new TextDecoder();
            let buffer = '';

            function pump() {
                return reader.read().then(({ done, value }) => {
                    if (done) {
                        return;
                    }

                    buffer += decoder.decode(value, { stream: true });
                    const lines = buffer.split('\n\n');
                    buffer = lines.pop() || '';

                    for (const line of lines) {
                        if (line.startsWith('data: ')) {
                            const dataStr = line.slice(6);
                            try {
                                const data = JSON.parse(dataStr);
                                if (data.type === 'done') {
                                    onDone && onDone(data);
                                } else if (data.type === 'error') {
                                    onDone && onDone({ success: false, message: data.message });
                                } else {
                                    onProgress && onProgress(data);
                                }
                            } catch (e) {
                                // 忽略
                            }
                        }
                    }

                    return pump();
                });
            }

            return pump();
        });
    },

    async getSubjects() {
        const response = await fetch(`${API_BASE}/subjects`);
        return response.json();
    },

    async getFiles(subject) {
        const url = subject
            ? `${API_BASE}/files?subject=${encodeURIComponent(subject)}`
            : `${API_BASE}/files`;
        const response = await fetch(url);
        return response.json();
    },
};
