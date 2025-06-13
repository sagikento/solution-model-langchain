import os
import json
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain.output_parsers import StructuredOutputParser, ResponseSchema

# .envファイルから環境変数を読み込む
load_dotenv()

# APIキーが設定されているか確認
if "GOOGLE_API_KEY" not in os.environ:
    raise ValueError("GOOGLE_API_KEYが設定されていません。.envファイルを確認してください。")

# 1. 出力形式のスキーマを定義
response_schemas = [
    ResponseSchema(
        name="solution_proposals",
        description="考えられる解決策の提案をリスト形式で返します。"
    ),
    ResponseSchema(
        name="similar_cases",
        description="過去の類似ケースや事例をリスト形式で返します。"
    ),
    ResponseSchema(
        name="cost_estimate",
        description="解決にかかる費用の相場をリスト形式で返します。"
    ),
]

# 2. LLMとパーサーを初期化
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0)
parser = StructuredOutputParser.from_response_schemas(response_schemas)

# 3. プロンプトテンプレートを作成
prompt = PromptTemplate(
    template="""あなたは優秀なコンサルタントです。
以下の課題について、解決策、類似ケース、費用相場を提案してください。

{format_instructions}

# 課題
## タイトル
{title}

## 詳細
{detail}
""",
    input_variables=["title", "detail"],
    partial_variables={"format_instructions": parser.get_format_instructions()},
)

# 4. チェーンを作成
chain = prompt | llm | parser

# 5. 実行と出力
def get_solution(title: str, detail: str):
    """
    課題のタイトルと詳細を入力し、解決策の提案をJSON形式で取得します。
    """
    try:
        response = chain.invoke({"title": title, "detail": detail})
        # Pythonの辞書をJSON文字列に変換して出力
        return json.dumps(response, indent=2, ensure_ascii=False)
    except Exception as e:
        return f"エラーが発生しました: {e}"

if __name__ == '__main__':
    # --- 入力 ---
    input_title = "社内の情報共有がうまくいかず、業務効率が低下している"
    input_detail = """
部署間の連携が不足しており、同じような資料を別々の部署で作成していることがある。
過去のプロジェクト資料やノウハウが個人のPCに保存されており、担当者が不在だと誰もアクセスできない。
新しいメンバーが入社しても、必要な情報を見つけるのに時間がかかっている。
"""

    # --- 実行 ---
    solution_json = get_solution(input_title, input_detail)

    # --- 出力 ---
    print(solution_json)