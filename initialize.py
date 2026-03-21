"""
このファイルは、最初の画面読み込み時にのみ実行される初期化処理が記述されたファイルです。
"""

############################################################
# ライブラリの読み込み
############################################################
import os
import logging
from collections import defaultdict
from logging.handlers import TimedRotatingFileHandler
from uuid import uuid4
import sys
import unicodedata
from dotenv import load_dotenv
import streamlit as st
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
import constants as ct


############################################################
# 設定関連
############################################################
# 「.env」ファイルで定義した環境変数の読み込み
load_dotenv()


############################################################
# 関数定義
############################################################

def initialize():
    """
    画面読み込み時に実行する初期化処理
    """
    # 初期化データの用意
    initialize_session_state()
    # ログ出力用にセッションIDを生成
    initialize_session_id()
    # ログ出力の設定
    initialize_logger()
    # RAGのRetrieverを作成
    initialize_retriever()



def initialize_logger():
    """
    ログ出力の設定
    """
    # 指定のログフォルダが存在すれば読み込み、存在しなければ新規作成
    os.makedirs(ct.LOG_DIR_PATH, exist_ok=True)

    # 引数に指定した名前のロガー（ログを記録するオブジェクト）を取得
    # 再度別の箇所で呼び出した場合、すでに同じ名前のロガーが存在していれば読み込む
    logger = logging.getLogger(ct.LOGGER_NAME)

    # すでにロガーにハンドラー（ログの出力先を制御するもの）が設定されている場合、同じログ出力が複数回行われないよう処理を中断する
    if logger.hasHandlers():
        return

    # 1日単位でログファイルの中身をリセットし、切り替える設定
    log_handler = TimedRotatingFileHandler(
        os.path.join(ct.LOG_DIR_PATH, ct.LOG_FILE),
        when="D",
        encoding="utf8"
    )
    # 出力するログメッセージのフォーマット定義
    # - 「levelname」: ログの重要度（INFO, WARNING, ERRORなど）
    # - 「asctime」: ログのタイムスタンプ（いつ記録されたか）
    # - 「lineno」: ログが出力されたファイルの行番号
    # - 「funcName」: ログが出力された関数名
    # - 「session_id」: セッションID（誰のアプリ操作か分かるように）
    # - 「message」: ログメッセージ
    formatter = logging.Formatter(
        f"[%(levelname)s] %(asctime)s line %(lineno)s, in %(funcName)s, session_id={st.session_state.session_id}: %(message)s"
    )

    # 定義したフォーマッターの適用
    log_handler.setFormatter(formatter)

    # ログレベルを「INFO」に設定
    logger.setLevel(logging.INFO)

    # 作成したハンドラー（ログ出力先を制御するオブジェクト）を、
    # ロガー（ログメッセージを実際に生成するオブジェクト）に追加してログ出力の最終設定
    logger.addHandler(log_handler)



def initialize_session_id():
    """
    セッションIDの作成
    """
    if "session_id" not in st.session_state:
        # ランダムな文字列（セッションID）を、ログ出力用に作成
        st.session_state.session_id = uuid4().hex



def initialize_retriever():
    """
    画面読み込み時にRAGのRetriever（ベクターストアから検索するオブジェクト）を作成
    """
    # すでにRetrieverが作成済みの場合、後続の処理を中断
    if "retriever" in st.session_state:
        return

    # RAGの参照先となるデータソースの読み込み
    docs_all = load_data_sources()

    # OSがWindowsの場合、Unicode正規化と、cp932（Windows用の文字コード）で表現できない文字を除去
    for doc in docs_all:
        doc.page_content = adjust_string(doc.page_content)
        for key in doc.metadata:
            doc.metadata[key] = adjust_string(doc.metadata[key])

    # 埋め込みモデルの用意
    embeddings = OpenAIEmbeddings()

    # ベクターストア登録用にドキュメントを整形
    vectorstore_docs = build_vectorstore_documents(docs_all)

    # ベクターストアの作成
    db = Chroma.from_documents(vectorstore_docs, embedding=embeddings)

    # ベクターストアを検索するRetrieverの作成
    st.session_state.retriever = db.as_retriever(search_kwargs={"k": ct.RAG_RETRIEVER_K})



def initialize_session_state():
    """
    初期化データの用意
    """
    if "messages" not in st.session_state:
        # 「表示用」の会話ログを順次格納するリストを用意
        st.session_state.messages = []
        # 「LLMとのやりとり用」の会話ログを順次格納するリストを用意
        st.session_state.chat_history = []



def load_data_sources():
    """
    RAGの参照先となるデータソースの読み込み

    Returns:
        読み込んだ通常データソース
    """
    # データソースを格納する用のリスト
    docs_all = []
    # ファイル読み込みの実行（渡した各リストにデータが格納される）
    recursive_file_check(ct.RAG_TOP_FOLDER_PATH, docs_all)

    web_docs_all = []
    # ファイルとは別に、指定のWebページ内のデータも読み込み
    # 読み込み対象のWebページ一覧に対して処理
    for web_url in ct.WEB_URL_LOAD_TARGETS:
        # 指定のWebページを読み込み
        loader = WebBaseLoader(web_url)
        web_docs = loader.load()
        # for文の外のリストに読み込んだデータソースを追加
        web_docs_all.extend(web_docs)
    # 通常読み込みのデータソースにWebページのデータを追加
    docs_all.extend(web_docs_all)

    return docs_all



def recursive_file_check(path, docs_all):
    """
    RAGの参照先となるデータソースの読み込み

    Args:
        path: 読み込み対象のファイル/フォルダのパス
        docs_all: データソースを格納する用のリスト
    """
    # パスがフォルダかどうかを確認
    if os.path.isdir(path):
        # フォルダの場合、フォルダ内のファイル/フォルダ名の一覧を取得
        files = os.listdir(path)
        # 各ファイル/フォルダに対して処理
        for file in files:
            # ファイル/フォルダ名だけでなく、フルパスを取得
            full_path = os.path.join(path, file)
            # フルパスを渡し、再帰的にファイル読み込みの関数を実行
            recursive_file_check(full_path, docs_all)
    else:
        # パスがファイルの場合、ファイル読み込み
        file_load(path, docs_all)



def file_load(path, docs_all):
    """
    ファイル内のデータ読み込み

    Args:
        path: ファイルパス
        docs_all: データソースを格納する用のリスト
    """
    # ファイルの拡張子を取得
    file_extension = os.path.splitext(path)[1].lower()

    # 想定していたファイル形式の場合のみ読み込む
    if file_extension in ct.SUPPORTED_EXTENSIONS:
        # ファイルの拡張子に合ったdata loaderを使ってデータ読み込み
        loader = ct.SUPPORTED_EXTENSIONS[file_extension](path)
        docs = loader.load()
        docs_all.extend(docs)



def build_vectorstore_documents(docs_all):
    """
    ベクターストアに登録するドキュメント一覧を作成

    Args:
        docs_all: 読み込んだ全ドキュメント

    Returns:
        ベクターストア登録用のドキュメント一覧
    """
    text_splitter = CharacterTextSplitter(
        chunk_size=ct.RAG_CHUNK_SIZE,
        chunk_overlap=ct.RAG_CHUNK_OVERLAP,
        separator="\n"
    )

    normal_docs = []
    csv_docs = []

    for doc in docs_all:
        source = doc.metadata.get("source", "")
        if isinstance(source, str) and source.lower().endswith(".csv"):
            csv_docs.append(doc)
        else:
            normal_docs.append(doc)

    splitted_docs = text_splitter.split_documents(normal_docs)
    aggregated_csv_docs = aggregate_csv_documents(csv_docs)

    return splitted_docs + aggregated_csv_docs



def aggregate_csv_documents(csv_docs):
    """
    CSVLoaderで1行ずつに分割されたCSVドキュメントを、ファイル単位の1ドキュメントに統合する

    Args:
        csv_docs: CSV由来のドキュメント一覧

    Returns:
        統合後のCSVドキュメント一覧
    """
    if not csv_docs:
        return []

    docs_by_source = defaultdict(list)
    for doc in csv_docs:
        source = doc.metadata.get("source", "")
        docs_by_source[source].append(doc)

    aggregated_docs = []

    for source, docs in docs_by_source.items():
        sorted_docs = sorted(docs, key=lambda doc: doc.metadata.get("row", 0))
        row_dicts = [parse_csv_row(doc.page_content) for doc in sorted_docs]
        csv_text = build_csv_search_text(source, row_dicts)
        aggregated_docs.append(Document(page_content=csv_text, metadata={"source": source}))

    return aggregated_docs



def parse_csv_row(page_content):
    """
    CSVLoaderが生成した1行分のテキストを辞書に変換する

    Args:
        page_content: CSVLoaderが生成した1行分のテキスト

    Returns:
        1行分の情報を格納した辞書
    """
    row_dict = {}

    for line in page_content.splitlines():
        if ":" not in line:
            continue

        key, value = line.split(":", 1)
        row_dict[key.strip()] = value.strip()

    return row_dict



def build_csv_search_text(source, row_dicts):
    """
    CSV検索精度向上用の統合テキストを作成する

    Args:
        source: CSVファイルのパス
        row_dicts: CSVの各行を辞書化したデータ

    Returns:
        ベクターストア登録用のテキスト
    """
    lines = [
        f"ファイルパス: {source}",
        "この文書はCSVファイル全体を1つに統合した検索用データです。",
        "各レコードは1件分の情報で、問い合わせ回答時には以下の情報を参照してください。"
    ]

    department_groups = defaultdict(list)
    for row_dict in row_dicts:
        department_name = get_csv_value(row_dict, ct.CSV_DEPARTMENT_KEYS, default_value="未分類")
        department_groups[department_name].append(row_dict)

    if department_groups:
        lines.append("\n【部署別インデックス】")
        for department_name, records in department_groups.items():
            employee_names = [name for name in [get_csv_value(record, ct.CSV_NAME_KEYS) for record in records] if name]
            joined_names = "、".join(employee_names)
            if joined_names:
                lines.append(f"- {department_name}: {len(records)}件（{joined_names}）")
            else:
                lines.append(f"- {department_name}: {len(records)}件")

    lines.append("\n【詳細データ】")
    for department_name, records in department_groups.items():
        lines.append(f"\n### {department_name}")
        for index, record in enumerate(records, start=1):
            lines.append(format_csv_record_text(record, index))

    return "\n".join(lines)



def get_csv_value(row_dict, candidate_keys, default_value=""):
    """
    候補キーの中から最初に一致した値を取得する

    Args:
        row_dict: CSV1行分の辞書
        candidate_keys: 値を探したいキー候補の一覧
        default_value: 一致しなかった場合の既定値

    Returns:
        一致した値。見つからない場合は既定値
    """
    for key in candidate_keys:
        if key in row_dict and row_dict[key]:
            return row_dict[key]

    for row_key, row_value in row_dict.items():
        if any(candidate_key in row_key for candidate_key in candidate_keys) and row_value:
            return row_value

    return default_value



def format_csv_record_text(record, index):
    """
    CSVの1レコード分を検索しやすいテキストに整形する

    Args:
        record: 1レコード分の辞書
        index: レコード番号

    Returns:
        整形後のテキスト
    """
    record_lines = [f"- レコード{index}"]
    for key, value in record.items():
        record_lines.append(f"  - {key}: {value}")

    return "\n".join(record_lines)



def adjust_string(s):
    """
    Windows環境でRAGが正常動作するよう調整

    Args:
        s: 調整を行う文字列

    Returns:
        調整を行った文字列
    """
    # 調整対象は文字列のみ
    if type(s) is not str:
        return s

    # OSがWindowsの場合、Unicode正規化と、cp932（Windows用の文字コード）で表現できない文字を除去
    if sys.platform.startswith("win"):
        s = unicodedata.normalize('NFC', s)
        s = s.encode("cp932", "ignore").decode("cp932")
        return s

    # OSがWindows以外の場合はそのまま返す
    return s
