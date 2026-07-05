#!/usr/bin/env python3
"""
RAG 知识库问答系统 - 零依赖版 Web 服务器

使用 Python 标准库 http.server 实现，无需安装任何第三方包。
支持：文档上传（TXT/MD）、知识库管理、问答对话。

运行方式（无需安装任何包）：
    python server_simple.py

然后在浏览器打开 http://localhost:8080
"""

import html
import json
import math
import os
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from typing import List, Optional
import tempfile
import shutil

# ============================================================
# 配置（修改这里或设置环境变量）
# ============================================================
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://api.deepseek.com/v1")
LLM_API_KEY = os.getenv("LLM_API_KEY", "")
LLM_MODEL = os.getenv("LLM_MODEL", "deepseek-chat")

EMBEDDING_BASE_URL = os.getenv("EMBEDDING_BASE_URL", LLM_BASE_URL)
EMBEDDING_API_KEY = os.getenv("EMBEDDING_API_KEY", LLM_API_KEY)
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")

DATA_DIR = Path("./data")
UPLOAD_DIR = DATA_DIR / "uploads"
VECTOR_DB_PATH = DATA_DIR / "vector_db.json"
CHUNK_SIZE = 400
CHUNK_OVERLAP = 50


# ============================================================
# 核心逻辑（内嵌，无外部依赖）
# ============================================================

def _cosine_similarity(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    return dot / (na * nb) if na and nb else 0.0


def _api_call(base_url, api_key, endpoint, payload):
    url = base_url.rstrip("/") + endpoint
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }, method="POST")
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.loads(resp.read().decode("utf-8"))


def load_txt(path):
    p = Path(path)
    try:
        return p.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return p.read_text(encoding="gbk")


def split_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    seps = ["\n\n", "\n", "。", "！", "？", ". ", "! ", "? ", " ", ""]
    def recurse(t, seps_left):
        if len(t) <= chunk_size:
            return [t] if t.strip() else []
        for i, sep in enumerate(seps_left):
            if sep == "":
                return [t[j:j+chunk_size] for j in range(0, len(t), chunk_size)]
            if sep in t:
                parts = t.split(sep)
                res = []
                joiner = sep if sep in "。！？" else ""
                for p in parts:
                    if len(p) <= chunk_size:
                        res.append(p + joiner if joiner else p)
                    else:
                        res.extend(recurse(p, seps_left[i+1:]))
                return [r for r in res if r.strip()]
        return [t]
    splits = recurse(text, seps)
    chunks, cur, clen = [], [], 0
    for s in splits:
        if clen + len(s) > chunk_size and cur:
            chunks.append("".join(cur))
            ov = chunks[-1][-overlap:] if overlap else ""
            cur, clen = [ov], len(ov)
        cur.append(s)
        clen += len(s)
    if cur:
        chunks.append("".join(cur))
    return [c for c in chunks if c.strip()]


def embed_texts(texts):
    if not EMBEDDING_API_KEY:
        raise ValueError("未设置 EMBEDDING_API_KEY")
    batch_size = 100
    all_vecs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        resp = _api_call(EMBEDDING_BASE_URL, EMBEDDING_API_KEY,
                         "/embeddings", {"input": batch, "model": EMBEDDING_MODEL})
        all_vecs.extend([d["embedding"] for d in sorted(resp["data"], key=lambda x: x["index"])])
    return all_vecs


def load_vector_db():
    if VECTOR_DB_PATH.exists():
        return json.loads(VECTOR_DB_PATH.read_text(encoding="utf-8"))
    return {"chunks": []}


def save_vector_db(data):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    VECTOR_DB_PATH.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def add_chunks_to_db(chunks, vectors, source):
    data = load_vector_db()
    start = len(data["chunks"])
    for i, (chunk, vec) in enumerate(zip(chunks, vectors)):
        data["chunks"].append({
            "id": f"chunk_{start+i}",
            "content": chunk,
            "source": source,
            "vector": vec,
        })
    save_vector_db(data)
    return len(vectors)


def search_db(query_vec, top_k=5):
    data = load_vector_db()
    if not data["chunks"]:
        return []
    scored = [(cosine_similarity(query_vec, item["vector"]), item) for item in data["chunks"]]
    scored.sort(key=lambda x: x[0], reverse=True)
    return [{"content": x[1]["content"], "source": x[1]["source"], "score": x[0]}
            for x in scored[:top_k]]


def ask_llm(query, context):
    if not LLM_API_KEY:
        return "请先设置 LLM_API_KEY 环境变量。"
    user_prompt = f"【参考资料】\n{context}\n\n【用户问题】\n{query}\n\n请根据以上参考资料回答。"
    system_prompt = """你是专业的知识库问答助手。根据【参考资料】回答。\n规则：1.只根据参考资料回答 2.没有相关内容请说明 3.回答末尾标注来源：[来源：文件名]"""
    resp = _api_call(LLM_BASE_URL, LLM_API_KEY, "/chat/completions", {
        "model": LLM_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.3, "max_tokens": 2048,
    })
    return resp["choices"][0]["message"]["content"]


# ============================================================
# HTTP 请求处理
# ============================================================

class RAGHandler(BaseHTTPRequestHandler):

    def _resp(self, status, data):
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", len(body))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def _resp_html(self, html_content):
        body = html_content.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", len(body))
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self):
        if self.path == "/" or self.path == "/index.html":
            self._resp_html(HTML_PAGE)
        elif self.path == "/api/status":
            data = load_vector_db()
            sources = sorted(set(item["source"] for item in data["chunks"]))
            self._resp(200, {"count": len(data["chunks"]), "sources": sources})
        else:
            self.send_error(404)

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length) if length else b"{}"

        if self.path == "/api/upload":
            self._handle_upload(body)
        elif self.path == "/api/chat":
            self._handle_chat(body)
        elif self.path == "/api/clear":
            data = {"chunks": []}
            save_vector_db(data)
            self._resp(200, {"ok": True, "msg": "知识库已清空"})
        else:
            self.send_error(404)

    def _handle_upload(self, body):
        try:
            form = urllib.parse.parse_qs(body.decode("utf-8"))
            content = form.get("content", [""])[0]
            filename = form.get("filename", ["upload.txt"])[0]

            if not content.strip():
                self._resp(400, {"error": "文件内容为空"})
                return

            chunks = split_text(content)
            if not chunks:
                self._resp(400, {"error": "文档切分后为空"})
                return

            # 如果没有 API Key，用模拟向量
            if not EMBEDDING_API_KEY or EMBEDDING_API_KEY == "sk-your-api-key-here":
                import random
                random.seed(42)
                dim = 10
                vectors = [[random.random() for _ in range(dim)] for _ in chunks]
                count = add_chunks_to_db(chunks, vectors, filename)
                self._resp(200, {"ok": True, "chunks": count,
                                 "msg": f"已添加 {count} 个文本块（模拟向量，请配置 API Key 启用真实检索）"})
            else:
                vectors = embed_texts(chunks)
                count = add_chunks_to_db(chunks, vectors, filename)
                self._resp(200, {"ok": True, "chunks": count,
                                 "msg": f"已添加 {count} 个文本块"})

        except Exception as e:
            self._resp(500, {"error": str(e)})

    def _handle_chat(self, body):
        try:
            data = json.loads(body)
            query = data.get("query", "").strip()
            if not query:
                self._resp(400, {"error": "请输入问题"})

            db = load_vector_db()
            if not db["chunks"]:
                self._resp(200, {"answer": "知识库为空，请先上传文档。", "sources": []})
                return

            # 如果没有真实向量（模拟数据），做简单关键词匹配
            if not EMBEDDING_API_KEY or EMBEDDING_API_KEY == "sk-your-api-key-here":
                # 简单关键词检索
                results = self._keyword_search(query, db, top_k=3)
                context = "\n\n".join(f"【参考{i+1}】{r['content']}" for i, r in enumerate(results))
                sources = list(set(r["source"] for r in results))
                answer = f"（模拟模式：已检索到 {len(results)} 个相关片段，请配置 API Key 获取 AI 回答）\n\n参考内容：\n{context[:500]}..."
            else:
                query_vec = embed_texts([query])[0]
                results = search_db(query_vec, top_k=5)
                context = "\n\n".join(f"【参考{i+1}】（来源：{r['source']}）\n{r['content']}" for i, r in enumerate(results))
                sources = list(set(r["source"] for r in results))
                answer = ask_llm(query, context)
                answer += f"\n\n---\n参考来源：{'、'.join(sources)}"

            self._resp(200, {"answer": answer, "sources": sources})

        except Exception as e:
            self._resp(500, {"error": str(e)})

    def _keyword_search(self, query, db, top_k=3):
        """简单关键词检索（无 API 时使用）"""
        words = set(query.lower().replace("？","").replace("，","").split())
        scored = []
        for item in db["chunks"]:
            content_lower = item["content"].lower()
            score = sum(1 for w in words if w in content_lower)
            if score > 0:
                scored.append((score, item))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [{"content": x[1]["content"], "source": x[1]["source"], "score": x[0]}
                for x in scored[:top_k]]


# ============================================================
# HTML 页面（内嵌）
# ============================================================

HTML_PAGE = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>RAG 知识库问答系统</title>
<style>
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: -apple-system, 'Segoe UI', sans-serif; background: #f5f5f5; }
.header { background: linear-gradient(135deg, #667eea, #764ba2); color: white; padding: 20px; text-align: center; }
.header h1 { font-size: 22px; font-weight: 600; }
.header p { font-size: 13px; opacity: 0.85; margin-top: 6px; }
.container { max-width: 960px; margin: 20px auto; padding: 0 16px; display: grid; grid-template-columns: 300px 1fr; gap: 16px; }
.card { background: white; border-radius: 12px; padding: 20px; box-shadow: 0 1px 4px rgba(0,0,0,0.08); }
.card h2 { font-size: 15px; font-weight: 600; margin-bottom: 14px; color: #333; }
#status { font-size: 13px; color: #666; line-height: 1.8; margin-bottom: 14px; }
#status span { color: #667eea; font-weight: 600; }
.btn { display: inline-block; padding: 8px 16px; border: none; border-radius: 8px; font-size: 13px; cursor: pointer; transition: all 0.2s; }
.btn-primary { background: #667eea; color: white; }
.btn-primary:hover { background: #5568d3; }
.btn-danger { background: #ef4444; color: white; }
.btn-danger:hover { background: #dc2626; }
.btn:disabled { opacity: 0.5; cursor: not-allowed; }
textarea, input[type=text] { width: 100%; padding: 10px; border: 1px solid #e0e0e0; border-radius: 8px; font-size: 13px; font-family: inherit; resize: vertical; }
textarea:focus, input[type=text]:focus { outline: none; border-color: #667eea; box-shadow: 0 0 0 2px rgba(102,126,234,0.15); }
#upload-area { border: 2px dashed #d0d0d0; border-radius: 10px; padding: 20px; text-align: center; color: #999; font-size: 13px; cursor: pointer; margin-bottom: 12px; transition: all 0.2s; }
#upload-area:hover { border-color: #667eea; color: #667eea; }
#file-input { display: none; }
#msg { font-size: 13px; padding: 10px; border-radius: 8px; margin-top: 10px; display: none; }
#msg.ok { background: #ecfdf5; color: #059669; }
#msg.err { background: #fef2f2; color: #dc2626; }
.chat-box { height: 420px; overflow-y: auto; border: 1px solid #e0e0e0; border-radius: 10px; padding: 16px; margin-bottom: 12px; background: #fafafa; }
.msg { margin-bottom: 14px; }
.msg-user { text-align: right; }
.msg-bubble { display: inline-block; max-width: 85%; padding: 10px 14px; border-radius: 12px; font-size: 13px; line-height: 1.6; }
.msg-user .msg-bubble { background: #667eea; color: white; }
.msg-ai .msg-bubble { background: white; color: #333; border: 1px solid #e8e8e8; }
.msg-source { font-size: 11px; color: #999; margin-top: 4px; padding-left: 4px; }
.input-row { display: flex; gap: 8px; }
.input-row input { flex: 1; }
#loading { display: none; font-size: 13px; color: #999; padding: 8px 0; }
#loading::after { content: ''; animation: dots 1.4s infinite; }
@keyframes dots { 0%{content: '';} 33%{content: '.';} 66%{content: '..';} 100%{content: '...';} }
</style>
</head>
<body>
<div class="header">
  <h1>RAG 知识库问答系统</h1>
  <p>上传文档构建知识库 · AI 基于文档内容回答</p>
</div>
<div class="container">
  <div class="left-col">
    <div class="card">
      <h2>知识库管理</h2>
      <div id="status">加载中...</div>
      <div id="upload-area" onclick="document.getElementById('file-input').click()">
        <div>点击上传文档</div>
        <div style="font-size:11px;margin-top:4px;">支持 TXT / MD（将内容粘贴上传）</div>
      </div>
      <input type="file" id="file-input" accept=".txt,.md" onchange="handleFile(event)">
      <div style="margin-bottom:12px;">
        <textarea id="paste-area" rows="5" placeholder="或直接粘贴文档内容到这里..."></textarea>
        <div style="margin-top:8px;display:flex;gap:8px;">
          <input type="text" id="filename-input" placeholder="文件名（可选）" value="粘贴文档.txt">
          <button class="btn btn-primary" onclick="handlePaste()">上传粘贴内容</button>
        </div>
      </div>
      <button class="btn btn-danger" onclick="clearKB()" style="width:100%;">清空知识库</button>
      <div id="msg"></div>
    </div>
  </div>
  <div class="right-col">
    <div class="card" style="height:100%;display:flex;flex-direction:column;">
      <h2>智能问答</h2>
      <div class="chat-box" id="chat-box">
        <div class="msg msg-ai">
          <div class="msg-bubble">你好！请先在左侧上传文档，然后向我提问，我会基于文档内容回答。</div>
        </div>
      </div>
      <div id="loading">AI 正在思考</div>
      <div class="input-row">
        <input type="text" id="query-input" placeholder="输入你的问题..." onkeydown="if(event.key==='Enter')sendQuery()">
        <button class="btn btn-primary" onclick="sendQuery()">发送</button>
      </div>
    </div>
  </div>
</div>
<script>
async function api(path, method='GET', body=null) {
  const opts = { method, headers: {'Content-Type': 'application/json'} };
  if (body) opts.body = JSON.stringify(body);
  const r = await fetch(path, opts);
  return r.json();
}

async function apiForm(path, body) {
  const r = await fetch(path, { method: 'POST', body });
  return r.json();
}

async function refreshStatus() {
  const d = await api('/api/status');
  document.getElementById('status').innerHTML =
    '<span>' + d.count + '</span> 个文本块<br>来源：' + (d.sources.length ? d.sources.join('、') : '（空）');
}

function showMsg(text, type) {
  const el = document.getElementById('msg');
  el.textContent = text;
  el.className = type;
  el.style.display = 'block';
  setTimeout(() => el.style.display = 'none', 4000);
}

async function handleFile(e) {
  const file = e.target.files[0];
  if (!file) return;
  const text = await file.text();
  await uploadContent(text, file.name);
}

async function handlePaste() {
  const text = document.getElementById('paste-area').value;
  const name = document.getElementById('filename-input').value || '粘贴文档.txt';
  if (!text.trim()) { showMsg('请先粘贴内容', 'err'); return; }
  await uploadContent(text, name);
}

async function uploadContent(content, filename) {
  const params = new URLSearchParams({ content, filename });
  const d = await apiForm('/api/upload', params);
  if (d.error) showMsg(d.error, 'err');
  else { showMsg(d.msg, 'ok'); document.getElementById('paste-area').value = ''; await refreshStatus(); }
}

async function clearKB() {
  if (!confirm('确定清空知识库？')) return;
  await api('/api/clear', 'POST');
  showMsg('知识库已清空', 'ok');
  await refreshStatus();
}

function addMessage(text, isUser) {
  const box = document.getElementById('chat-box');
  const div = document.createElement('div');
  div.className = 'msg ' + (isUser ? 'msg-user' : 'msg-ai');
  div.innerHTML = '<div class="msg-bubble">' + text.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/\n/g,'<br>') + '</div>';
  box.appendChild(div);
  box.scrollTop = box.scrollHeight;
  return div;
}

async function sendQuery() {
  const input = document.getElementById('query-input');
  const query = input.value.trim();
  if (!query) return;
  input.value = '';
  addMessage(query, true);
  const loading = document.getElementById('loading');
  loading.style.display = 'block';
  try {
    const d = await api('/api/chat', 'POST', { query });
    loading.style.display = 'none';
    let answer = d.answer || d.error || '未知错误';
    let sources = d.sources || [];
    const div = addMessage(answer, false);
    if (sources.length) {
      const src = document.createElement('div');
      src.className = 'msg-source';
      src.textContent = '参考来源：' + sources.join('、');
      div.appendChild(src);
    }
  } catch(e) {
    loading.style.display = 'none';
    addMessage('请求失败：' + e.message, false);
  }
}

refreshStatus();
</script>
</body>
</html>""".encode("utf-8").decode("utf-8")


def main():
    PORT = 8080
    server = HTTPServer(("0.0.0.0", PORT), RAGHandler)
    print(f"\n{'='*50}")
    print(f"  RAG 知识库问答系统启动")
    print(f"  浏览器打开: http://localhost:{PORT}")
    print(f"  无需安装任何依赖！")
    print(f"  按 Ctrl+C 停止")
    print(f"{'='*50}\n")
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n已停止。")


if __name__ == "__main__":
    main()
