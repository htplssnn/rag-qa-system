#!/usr/bin/env python3
"""
RAG 绯荤粺闆朵緷璧栨祴璇�

涓嶉渶瑕佸畨瑁呬换浣曠涓夋柟鍖咃紒
- 娴嬭瘯鏂囨。鍔犺浇锛圱XT锛夈€佹枃鏈垏鍒嗐€佸悜閲忓瓨鍌紙JSON鏂囦欢锛�
- 濡傛灉璁剧疆浜� API Key锛岃繕娴嬭瘯鐪熷疄 Embedding + LLM 璋冪敤

杩愯鏂瑰紡锛堟棤闇€瀹夎浠讳綍鍖咃級锛�
    python test_simple.py

鐜鍙橀噺锛堝彲閫夛紝鐢ㄤ簬娴嬭瘯瀹屾暣娴佺▼锛夛細
    LLM_API_KEY      澶фā鍨� API Key
    LLM_BASE_URL     API 鍦板潃锛堥粯璁� DeepSeek锛�
    LLM_MODEL        妯″瀷鍚嶏紙榛樿 deepseek-chat锛�
"""

import json
import math
import os
import sys
import tempfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path

# ============================================================
# 闆朵緷璧� RAG 鏍稿績锛堝唴宓岋紝涓嶄緷璧栦换浣曠涓夋柟鍖咃級
# ============================================================

SYSTEM_PROMPT = """浣犳槸涓撲笟鐨勭煡璇嗗簱闂瓟鍔╂墜銆傛牴鎹€愬弬鑰冭祫鏂欍€戝洖绛斻€�

瑙勫垯锛�
1. 鍙牴鎹弬鑰冭祫鏂欏洖绛旓紝涓嶇紪閫犮€�
2. 娌℃湁鐩稿叧鍐呭璇疯鏄庛€�
3. 鍥炵瓟鏈熬鏍囨敞鏉ユ簮锛歔鏉ユ簮锛氭枃浠跺悕]"""


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
    """鍔犺浇 TXT/MD 鏂囦欢"""
    p = Path(path)
    try:
        return p.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return p.read_text(encoding="gbk")


def split_text(text, chunk_size=300, overlap=40):
    """閫掑綊瀛楃鍒囧垎锛堢函 Python 瀹炵幇锛�"""
    seps = ["\n\n", "\n", "銆�", "锛�", "锛�", ". ", "! ", "? ", " ", ""]
    def recurse(t, seps_left):
        if len(t) <= chunk_size:
            return [t] if t.strip() else []
        for i, sep in enumerate(seps_left):
            if sep == "":
                return [t[j:j+chunk_size] for j in range(0, len(t), chunk_size)]
            if sep in t:
                parts = t.split(sep)
                res = []
                joiner = sep if sep in "銆傦紒锛�" else ""
                for p in parts:
                    if len(p) <= chunk_size:
                        res.append(p + joiner if joiner else p)
                    else:
                        res.extend(recurse(p, seps_left[i+1:]))
                return [r for r in res if r.strip()]
        return [t]
    splits = recurse(text, seps)
    # 鍚堝苟鎴� chunk_size 澶у皬鐨勫潡
    chunks, cur, clen = [], [], 0
    for s in splits:
        if clen + len(s) > chunk_size and cur:
            chunks.append("".join(cur))
            overlap_text = chunks[-1][-overlap:] if overlap else ""
            cur, clen = [overlap_text], len(overlap_text)
        cur.append(s)
        clen += len(s)
    if cur:
        chunks.append("".join(cur))
    return [c for c in chunks if c.strip()]


def embed_texts(texts, base_url, api_key, model="text-embedding-3-small"):
    """璋冪敤 Embedding API"""
    batch_size = 100
    all_vecs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        resp = _api_call(base_url, api_key, "/embeddings", {
            "input": batch, "model": model
        })
        all_vecs.extend([d["embedding"] for d in sorted(resp["data"], key=lambda x: x["index"])])
    return all_vecs


def save_vectors(db_path, chunks, vectors, sources):
    """淇濆瓨鍚戦噺鍒� JSON 鏂囦欢"""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    data = {"chunks": []}
    if db_path.exists():
        data = json.loads(db_path.read_text(encoding="utf-8"))
    start = len(data["chunks"])
    for i, (chunk, vec) in enumerate(zip(chunks, vectors)):
        data["chunks"].append({
            "id": f"chunk_{start+i}",
            "content": chunk,
            "source": sources.get(i, "鏈煡"),
            "vector": vec,
        })
    db_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return len(vectors)


def search_vectors(db_path, query_vec, top_k=5):
    """妫€绱㈡渶鐩镐技鐨勬枃鏈潡"""
    if not db_path.exists():
        return []
    data = json.loads(db_path.read_text(encoding="utf-8"))
    scored = []
    for item in data["chunks"]:
        score = _cosine_similarity(query_vec, item["vector"])
        scored.append((score, item))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [{"content": x[1]["content"], "source": x[1]["source"], "score": x[0]}
            for x in scored[:top_k]]


def ask_llm(query, context, base_url, api_key, model="deepseek-chat"):
    """璋冪敤澶фā鍨嬬敓鎴愬洖绛�"""
    user_prompt = f"銆愬弬鑰冭祫鏂欍€慭n{context}\n\n銆愮敤鎴烽棶棰樸€慭n{query}\n\n璇锋牴鎹互涓婂弬鑰冭祫鏂欏洖绛斻€�"
    resp = _api_call(base_url, api_key, "/chat/completions", {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.3,
        "max_tokens": 2048,
    })
    return resp["choices"][0]["message"]["content"]


# ============================================================
# 娴嬭瘯閫昏緫
# ============================================================

def separator(title):
    print(f"\n{'='*54}")
    print(f"  {title}")
    print(f"{'='*54}")


def test_offline():
    """绂荤嚎娴嬭瘯锛氫笉渚濊禆 API锛屾祴璇曟暟鎹粨鏋勫拰鍒囧垎閫昏緫"""
    separator("绂荤嚎娴嬭瘯锛堟棤闇€ API Key锛�")

    # 1. 鍒涘缓娴嬭瘯鏂囨。
    test_content = """鏅鸿兘鍔╂墜锛圛ntelligent Assistant锛夋槸涓€娆句汉宸ユ櫤鑳藉姪鎵嬩骇鍝侊紝
鑳藉甯姪鐢ㄦ埛瀹屾垚澶氱浠诲姟锛屽寘鎷俊鎭绱€€佹枃鏈敓鎴愩€佷唬鐮佺紪鍐欑瓑銆�

RAG锛堟绱㈠寮虹敓鎴愶級鎶€鏈€氳繃灏嗗閮ㄧ煡璇嗗簱涓庡ぇ鍨嬭瑷€妯″瀷鐩哥粨鍚堬紝
鏄捐憲鎻愬崌浜嗘ā鍨嬪洖绛旂殑鍑嗙‘鎬у拰鏃舵晥鎬с€�

鍚戦噺鏁版嵁搴撴槸 RAG 绯荤粺鐨勬牳蹇冪粍浠朵箣涓€锛屽父瑙佺殑鍚戦噺鏁版嵁搴撳寘鎷細
ChromaDB銆丳inecone銆乄eaviate銆丵drant 绛夈€�

鏂囨湰鍒囧垎绛栫暐瀵� RAG 鏁堟灉鏈夐噸瑕佸奖鍝嶃€�
chunk_size 杩囧皬浼氫涪澶变笂涓嬫枃锛岃繃澶у垯寮曞叆鍣０銆�"""

    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8") as f:
        f.write(test_content)
        tmp_path = f.name

    try:
        # 2. 娴嬭瘯鏂囨。鍔犺浇
        content = load_txt(tmp_path)
        print(f"  [1/4] 鏂囨。鍔犺浇: {len(content)} 瀛楃 [OK]")

        # 3. 娴嬭瘯鏂囨湰鍒囧垎
        chunks = split_text(content, chunk_size=150, overlap=30)
        print(f"  [2/4] 鏂囨湰鍒囧垎: {len(chunks)} 涓潡 [OK]")
        for i, c in enumerate(chunks):
            print(f"        鍧梴i+1} ({len(c)}瀛�): {c[:40]}...")

        # 4. 娴嬭瘯鍚戦噺瀛樺偍锛堢敤闅忔満鍚戦噺妯℃嫙锛�
        print(f"  [3/4] 鍚戦噺瀛樺偍娴嬭瘯锛堟ā鎷熷悜閲忥級...")
        db_path = Path("./data/test_vector_db.json")
        db_path.parent.mkdir(parents=True, exist_ok=True)

        # 鐢熸垚妯℃嫙鍚戦噺锛堢敤瀛楃棰戠巼浣滀负绠€鍗曠壒寰侊級
        import random
        random.seed(42)
        fake_vectors = [[random.random() for _ in range(10)] for _ in chunks]

        data = {"chunks": []}
        for i, (chunk, vec) in enumerate(zip(chunks, fake_vectors)):
            data["chunks"].append({
                "id": f"test_{i}",
                "content": chunk,
                "source": "test.txt",
                "vector": vec,
            })
        db_path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        print(f"        宸蹭繚瀛� {len(chunks)} 鏉℃ā鎷熷悜閲忓埌 {db_path}")

        # 5. 娴嬭瘯妫€绱紙鐢ㄧ涓€涓潡鐨勫悜閲忔煡鑷繁锛�
        query_vec = fake_vectors[0]
        results = search_vectors(db_path, query_vec, top_k=2)
        print(f"  [4/4] 鍚戦噺妫€绱㈡祴璇�: 杩斿洖 {len(results)} 鏉＄粨鏋� [OK]")
        for r in results:
            print(f"        鐩镐技搴� {r['score']:.4f}: {r['content'][:40]}...")

        db_path.unlink(missing_ok=True)
        print(f"\n  [PASS] 绂荤嚎娴嬭瘯鍏ㄩ儴閫氳繃锛�")

    finally:
        os.unlink(tmp_path)


def test_online():
    """鍦ㄧ嚎娴嬭瘯锛氶渶瑕� API Key"""
    separator("鍦ㄧ嚎娴嬭瘯锛堥渶瑕� API Key锛�")

    api_key = os.getenv("LLM_API_KEY", "")
    if not api_key or api_key == "sk-your-api-key-here":
        print("  [SKIP] 鏈缃� LLM_API_KEY锛岃烦杩囧湪绾挎祴璇�")
        print("  璁剧疆鏂规硶: set LLM_API_KEY=your-key")
        return False

    base_url = os.getenv("LLM_BASE_URL", "https://api.deepseek.com/v1")
    model = os.getenv("LLM_MODEL", "deepseek-chat")
    embedding_model = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")

    print(f"  API: {base_url}")
    print(f"  妯″瀷: {model}")

    # 1. 鍑嗗娴嬭瘯鏂囨。
    test_content = """RAG锛圧etrieval-Augmented Generation锛屾绱㈠寮虹敓鎴愶級鏄竴绉嶇粨鍚堜俊鎭绱㈠拰鏂囨湰鐢熸垚鐨勬妧鏈€�

RAG 鐨勫伐浣滄祦绋嬪垎涓轰袱姝ワ細
绗竴姝ワ細鏍规嵁鐢ㄦ埛闂锛屼粠鐭ヨ瘑搴撲腑妫€绱㈡渶鐩稿叧鐨勬枃妗ｇ墖娈点€�
绗簩姝ワ細灏嗘绱㈠埌鐨勬枃妗ｇ墖娈典綔涓轰笂涓嬫枃锛屼氦缁欏ぇ璇█妯″瀷鐢熸垚鍥炵瓟銆�

RAG 鐨勪富瑕佷紭鍔匡細
1. 鍑忓皯骞昏锛氭ā鍨嬪彧鍩轰簬妫€绱㈠埌鐨勭湡瀹炴枃妗ｅ洖绛旓紝鍑忓皯缂栭€犮€�
2. 鐭ヨ瘑鍙洿鏂帮細鍙渶鏇存柊鐭ヨ瘑搴擄紝鏃犻渶閲嶆柊璁粌妯″瀷銆�
3. 鍙函婧愶細鍥炵瓟鍙互鏍囨敞淇℃伅鏉ユ簮锛屾彁楂樺彲淇″害銆�

甯歌鐨� RAG 浼樺寲鏂瑰悜锛�
- 娣峰悎妫€绱細缁撳悎鍚戦噺妫€绱㈠拰鍏抽敭璇嶆绱紙濡� BM25锛夈€�
- Re-ranking锛氬妫€绱㈢粨鏋滈噸鏂版帓搴忥紝鎻愬崌绮惧害銆�
- 澶氳疆妫€绱細鏍规嵁瀵硅瘽鍘嗗彶杩涜澶氭妫€绱€€�"""

    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8") as f:
        f.write(test_content)
        tmp_path = f.name

    try:
        # 2. 鍒囧垎
        print(f"\n  [1/5] 鍒囧垎鏂囨。...")
        chunks = split_text(load_txt(tmp_path), chunk_size=200, overlap=30)
        print(f"        寰楀埌 {len(chunks)} 涓枃鏈潡")

        # 3. 鍚戦噺鍖�
        print(f"  [2/5] 璋冪敤 Embedding API...")
        vectors = embed_texts(chunks, base_url, api_key, embedding_model)
        print(f"        寰楀埌 {len(vectors)} 涓悜閲忥紝缁村害 {len(vectors[0])}")

        # 4. 瀛樺偍
        db_path = Path("./data/vector_db.json")
        save_vectors(db_path, chunks, vectors, {i: "test_rag.txt" for i in range(len(chunks))})
        print(f"  [3/5] 宸蹭繚瀛樺埌 {db_path}")

        # 5. 妫€绱� + 闂瓟
        query = "RAG 鐨勪富瑕佷紭鍔挎湁鍝簺锛�"
        print(f"\n  [4/5] 妫€绱�: 銆寋query}銆�")
        query_vec = embed_texts([query], base_url, api_key, embedding_model)[0]
        results = search_vectors(db_path, query_vec, top_k=3)

        context_parts = []
        for i, r in enumerate(results):
            print(f"        缁撴灉{i+1} [鐩镐技搴� {r['score']:.4f}]: {r['content'][:50]}...")
            context_parts.append(f"銆愬弬鑰儃i+1}銆憑r['content']}")
        context = "\n\n".join(context_parts)

        # 6. 鐢熸垚鍥炵瓟
        print(f"  [5/5] 璋冪敤澶фā鍨嬬敓鎴愬洖绛�...")
        answer = ask_llm(query, context, base_url, api_key, model)

        print(f"\n  闂: {query}")
        print(f"  鍥炵瓟:\n{answer}")
        print(f"\n  [PASS] 鍦ㄧ嚎娴嬭瘯鍏ㄩ儴閫氳繃锛�")

        # 娓呯悊
        db_path.unlink(missing_ok=True)

    except Exception as e:
        print(f"\n  [FAIL] 鍦ㄧ嚎娴嬭瘯澶辫触: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        os.unlink(tmp_path)

    return True


def main():
    print("\n" + "=" * 54)
    print("  RAG 鐭ヨ瘑搴撻棶绛旂郴缁� - 闆朵緷璧栨祴璇�")
    print("=" * 54)

    test_offline()
    test_online()

    separator("娴嬭瘯瀹屾垚")
    print("  绂荤嚎娴嬭瘯锛氶獙璇佹暟鎹粨鏋勫拰鍒囧垎閫昏緫 [OK]")
    print("  鍦ㄧ嚎娴嬭瘯锛氶獙璇佸畬鏁� RAG 娴佺▼锛堥渶 API Key锛�")
    print()
    print("  涓嬩竴姝ワ細")
    print("    1. 璁剧疆 API Key: set LLM_API_KEY=your-key")
    print("    2. 杩愯瀹屾暣娴嬭瘯: python test_simple.py")
    print("    3. 鍚姩 Web 鐣岄潰: python app_simple.py")
    print()


if __name__ == "__main__":
    main()
