"""
このファイルは、画面表示に特化した関数定義のファイルです。
"""

############################################################
# ライブラリの読み込み
############################################################
import streamlit as st
import utils
import constants as ct


############################################################
# 関数定義
############################################################

def display_app_title():
    """
    タイトル表示
    """
    st.markdown(f"## {ct.APP_NAME}")



def display_select_mode():
    """
    回答モードのラジオボタンをサイドバーに表示
    """
    with st.sidebar:
        st.markdown(f"### {ct.SIDEBAR_TITLE}")
        st.session_state.mode = st.radio(
            label="",
            options=[ct.ANSWER_MODE_1, ct.ANSWER_MODE_2],
            label_visibility="collapsed"
        )
        st.divider()

        st.markdown(f"**【{ct.ANSWER_MODE_1}を選択した場合】**")
        st.info(ct.DOC_SEARCH_DESCRIPTION)
        st.code(ct.DOC_SEARCH_EXAMPLE, wrap_lines=True, language=None)

        st.markdown(f"**【{ct.ANSWER_MODE_2}を選択した場合】**")
        st.info(ct.INQUIRY_DESCRIPTION)
        st.code(ct.INQUIRY_EXAMPLE, wrap_lines=True, language=None)



def display_initial_ai_message():
    """
    初期案内メッセージの表示
    """
    st.success(ct.INITIAL_ASSISTANT_MESSAGE, icon=ct.DOC_SOURCE_ICON)
    st.warning(ct.INITIAL_WARNING_MESSAGE, icon=ct.WARNING_ICON)



def display_source_box(source_info, message_type="info"):
    """
    参照元情報を画面表示する

    Args:
        source_info: 参照元情報の辞書
        message_type: streamlitで使う表示形式
    """
    icon = utils.get_source_icon(source_info["source"])
    display_text = source_info["display_text"]

    if message_type == "success":
        st.success(display_text, icon=icon)
    else:
        st.info(display_text, icon=icon)



def display_conversation_log():
    """
    会話ログの一覧表示
    """
    # 会話ログのループ処理
    for message in st.session_state.messages:
        # 「message」辞書の中の「role」キーには「user」か「assistant」が入っている
        with st.chat_message(message["role"]):

            # ユーザー入力値の場合、そのままテキストを表示するだけ
            if message["role"] == "user":
                st.markdown(message["content"])

            # LLMからの回答の場合
            else:
                # 「社内文書検索」の場合、テキストの種類に応じて表示形式を分岐処理
                if message["content"]["mode"] == ct.ANSWER_MODE_1:

                    # ファイルのありかの情報が取得できた場合（通常時）の表示処理
                    if "no_file_path_flg" not in message["content"]:
                        # ==========================================
                        # ユーザー入力値と最も関連性が高いメインドキュメントのありかを表示
                        # ==========================================
                        st.markdown(message["content"]["main_message"])
                        display_source_box(message["content"]["main_source_info"], message_type="success")

                        # ==========================================
                        # ユーザー入力値と関連性が高いサブドキュメントのありかを表示
                        # ==========================================
                        if "sub_message" in message["content"]:
                            st.markdown(message["content"]["sub_message"])

                            for sub_choice in message["content"]["sub_choices"]:
                                display_source_box(sub_choice)
                    # ファイルのありかの情報が取得できなかった場合、LLMからの回答のみ表示
                    else:
                        st.markdown(message["content"]["answer"])

                # 「社内問い合わせ」の場合の表示処理
                else:
                    # LLMからの回答を表示
                    st.markdown(message["content"]["answer"])

                    # 参照元のありかを一覧表示
                    if "file_info_list" in message["content"]:
                        st.divider()
                        st.markdown(f"##### {message['content']['message']}")
                        for file_info in message["content"]["file_info_list"]:
                            display_source_box(file_info)



def display_search_llm_response(llm_response):
    """
    「社内文書検索」モードにおけるLLMレスポンスを表示

    Args:
        llm_response: LLMからの回答

    Returns:
        LLMからの回答を画面表示用に整形した辞書データ
    """
    # LLMからのレスポンスに参照元情報が入っており、かつ「該当資料なし」が回答として返された場合
    if llm_response["context"] and llm_response["answer"] != ct.NO_DOC_MATCH_ANSWER:

        # ==========================================
        # ユーザー入力値と最も関連性が高いメインドキュメントのありかを表示
        # ==========================================
        main_document = llm_response["context"][0]
        main_file_path = main_document.metadata["source"]
        main_page_number = main_document.metadata.get("page")
        main_source_info = utils.build_source_info(main_file_path, main_page_number)

        main_message = "入力内容に関する情報は、以下のファイルに含まれている可能性があります。"
        st.markdown(main_message)
        display_source_box(main_source_info, message_type="success")

        # ==========================================
        # ユーザー入力値と関連性が高いサブドキュメントのありかを表示
        # ==========================================
        sub_choices = []
        duplicate_check_list = [main_file_path]

        for document in llm_response["context"][1:]:
            sub_file_path = document.metadata["source"]

            if sub_file_path in duplicate_check_list:
                continue

            duplicate_check_list.append(sub_file_path)
            sub_page_number = document.metadata.get("page")
            sub_choice = utils.build_source_info(sub_file_path, sub_page_number)
            sub_choices.append(sub_choice)

        if sub_choices:
            sub_message = "その他、ファイルありかの候補を提示します。"
            st.markdown(sub_message)

            for sub_choice in sub_choices:
                display_source_box(sub_choice)

        content = {
            "mode": ct.ANSWER_MODE_1,
            "main_message": main_message,
            "main_source_info": main_source_info
        }
        if sub_choices:
            content["sub_message"] = sub_message
            content["sub_choices"] = sub_choices

    # LLMからのレスポンスに、ユーザー入力値と関連性の高いドキュメント情報が入って「いない」場合
    else:
        st.markdown(ct.NO_DOC_MATCH_MESSAGE)
        content = {
            "mode": ct.ANSWER_MODE_1,
            "answer": ct.NO_DOC_MATCH_MESSAGE,
            "no_file_path_flg": True
        }

    return content



def display_contact_llm_response(llm_response):
    """
    「社内問い合わせ」モードにおけるLLMレスポンスを表示

    Args:
        llm_response: LLMからの回答

    Returns:
        LLMからの回答を画面表示用に整形した辞書データ
    """
    st.markdown(llm_response["answer"])

    if llm_response["answer"] != ct.INQUIRY_NO_MATCH_ANSWER:
        st.divider()

        message = "情報源"
        st.markdown(f"##### {message}")

        file_path_list = []
        file_info_list = []

        for document in llm_response["context"]:
            file_path = document.metadata["source"]
            if file_path in file_path_list:
                continue

            source_info = utils.build_source_info(file_path, document.metadata.get("page"))
            display_source_box(source_info)

            file_path_list.append(file_path)
            file_info_list.append(source_info)

    content = {
        "mode": ct.ANSWER_MODE_2,
        "answer": llm_response["answer"]
    }
    if llm_response["answer"] != ct.INQUIRY_NO_MATCH_ANSWER:
        content["message"] = message
        content["file_info_list"] = file_info_list

    return content
