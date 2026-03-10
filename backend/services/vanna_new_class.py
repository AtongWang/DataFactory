from vanna.base import VannaBase
import re
import ast
import plotly
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import traceback
import logging

logger = logging.getLogger(__name__)


class EnhancedVannaBase(VannaBase):
    """
    增强的VannaBase类，用于解决plotly代码生成和解析问题
    特别是处理LLM生成函数定义而非直接图表变量的情况
    """

    def log(self, message, title=None):
        if title:
            log_message = f"{title}: {message}"
        else:
            log_message = message
        return logger.info(log_message)

    def _clean_thinking_tags(self, response):
        """移除LLM响应中的<think>...</think>标签"""
        return re.sub(r"<think>[\s\S]*?</think>", "", response).strip()

    def _extract_python_code(self, markdown_string: str) -> str:
        """从文本中提取Python代码，去除markdown格式和思考标签"""
        markdown_string = self._clean_thinking_tags(markdown_string)
        # Strip whitespace to avoid indentation errors in LLM-generated code
        markdown_string = markdown_string.strip()

        # Regex pattern to match Python code blocks
        pattern = r"```[\w\s]*python\n([\s\S]*?)```|```([\s\S]*?)```"

        # Find all matches in the markdown string
        matches = re.findall(pattern, markdown_string, re.IGNORECASE)

        # Extract the Python code from the matches
        python_code = []
        for match in matches:
            python = match[0] if match[0] else match[1]
            python_code.append(python.strip())

        if len(python_code) == 0:
            return markdown_string

        return python_code[0]

    # 抛弃原方法 基于 原来方法
    def _extract_python_code(self, code_str: str) -> str:
        """从文本中提取Python代码，去除markdown格式和思考标签"""
        # 先清理思考标签
        code_str = self._clean_thinking_tags(code_str)

        # 尝试提取markdown代码块
        code_match = re.search(r"```(?:python)?\s*([\s\S]*?)\s*```", code_str)
        if code_match:
            return code_match.group(1).strip()

        # 如果没有markdown格式，返回整个字符串
        return code_str.strip()

    def _sanitize_plotly_code(self, code: str) -> str:
        """清理和优化plotly代码"""
        # 确保代码中没有思考标签
        code = self._clean_thinking_tags(code)
        code = re.sub(r"\b\w+\.show\s*\([^\)]*\)", "", code)

        # 确保有导入语句
        if "import plotly" not in code and "from plotly" not in code:
            code = (
                "import plotly.graph_objects as go\nimport plotly.express as px\n"
                + code
            )

        # 确保代码创建了fig变量
        if "fig =" not in code and "fig=" not in code:
            self.log("警告：代码中未找到fig变量定义")

        return code

    def get_plotly_figure(
        self, plotly_code: str, df: pd.DataFrame, dark_mode: bool = True
    ) -> plotly.graph_objs.Figure:
        """
        增强版的get_plotly_figure方法，能够处理多种形式的plotly代码

        Args:
            plotly_code (str): 由LLM生成的plotly代码
            df (pd.DataFrame): 数据框
            dark_mode (bool): 是否使用深色模式

        Returns:
            plotly.graph_objs.Figure: plotly图表对象
        """
        # 确保代码不包含思考标签
        plotly_code = self._clean_thinking_tags(plotly_code)
        plotly_code = self._sanitize_plotly_code(plotly_code)

        ldict = {"df": df, "px": px, "go": go, "pd": pd}

        try:
            # 执行代码
            exec(plotly_code, globals(), ldict)

            # 1. 检查是否直接创建了fig变量
            fig = ldict.get("fig", None)
            if fig is not None:
                if dark_mode:
                    fig.update_layout(template="plotly_dark")
                return fig

            # 2. 检查是否定义了返回图表的函数
            function_names = []
            tree = ast.parse(plotly_code)
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    function_names.append(node.name)

            # 尝试调用找到的函数
            for func_name in function_names:
                if func_name in ldict:
                    try:
                        # 尝试不带参数调用
                        fig = ldict[func_name]()
                        if isinstance(fig, plotly.graph_objs.Figure):
                            if dark_mode:
                                fig.update_layout(template="plotly_dark")
                            return fig
                    except TypeError:
                        try:
                            # 尝试传入df参数调用
                            fig = ldict[func_name](df)
                            if isinstance(fig, plotly.graph_objs.Figure):
                                if dark_mode:
                                    fig.update_layout(template="plotly_dark")
                                return fig
                        except Exception:
                            pass

            # 3. 如果以上方法都失败，使用备选方案
            self.log("未找到有效的plotly图表对象，使用备选方案创建图表")
            return self._create_fallback_figure(df, dark_mode)

        except Exception as e:
            self.log(f"执行plotly代码时出错: {str(e)}")
            traceback.print_exc()
            return self._create_fallback_figure(df, dark_mode)

    def _create_fallback_figure(
        self, df: pd.DataFrame, dark_mode: bool = True
    ) -> plotly.graph_objs.Figure:
        """
        当无法从LLM生成的代码中获取图表时，创建备选图表

        Args:
            df (pd.DataFrame): 数据框
            dark_mode (bool): 是否使用深色模式

        Returns:
            plotly.graph_objs.Figure: 备选plotly图表
        """
        # 检查数据框中的列类型
        numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
        categorical_cols = df.select_dtypes(
            include=["object", "category"]
        ).columns.tolist()

        # 根据数据特征选择适当的图表类型
        fig = None

        if len(df) == 1 and len(numeric_cols) >= 1:
            # 单一值使用指标图
            value = df.iloc[0, df.columns.get_loc(numeric_cols[0])]
            fig = go.Figure(
                go.Indicator(
                    mode="number", value=value, title={"text": numeric_cols[0]}
                )
            )
        elif len(numeric_cols) >= 2:
            # 两个或更多数值列使用散点图
            fig = px.scatter(df, x=numeric_cols[0], y=numeric_cols[1])
        elif len(numeric_cols) == 1 and len(categorical_cols) >= 1:
            # 一个数值列和一个分类列使用条形图
            fig = px.bar(df, x=categorical_cols[0], y=numeric_cols[0])
        elif len(categorical_cols) >= 1 and df[categorical_cols[0]].nunique() < 10:
            # 对于分类数据，如果唯一值不多，使用饼图
            fig = px.pie(df, names=categorical_cols[0])
        else:
            # 默认使用表格视图
            fig = go.Figure(
                data=[
                    go.Table(
                        header=dict(values=list(df.columns)),
                        cells=dict(values=[df[col] for col in df.columns]),
                    )
                ]
            )

        if dark_mode:
            fig.update_layout(template="plotly_dark")

        return fig

    # def generate_plotly_code(self, question: str = None, sql: str = None, df_metadata: str = None, **kwargs) -> str:
    #     """
    #     增强版的generate_plotly_code方法，生成更直接有效的plotly代码

    #     Args:
    #         question (str): 用户问题
    #         sql (str): SQL查询
    #         df_metadata (str): 数据框元数据

    #     Returns:
    #         str: plotly代码
    #     """
    #     if question is not None:
    #         system_msg = f"The following is a pandas DataFrame that contains the results of the query that answers the question the user asked: '{question}'"
    #     else:
    #         system_msg = "The following is a pandas DataFrame "

    #     if sql is not None:
    #         system_msg += f"\n\nThe DataFrame was produced using this query: {sql}\n\n"

    #     system_msg += f"The following is information about the resulting pandas DataFrame 'df': \n{df_metadata}"

    #     message_log = [
    #         self.system_message(system_msg),
    #         self.user_message(
    #             "Generate Python plotly code to visualize this dataframe. The code should directly create a plotly figure variable named 'fig'. DO NOT define functions, just write code that directly creates the 'fig' variable. If there is only one value in the dataframe, use an Indicator. Respond with ONLY Python code, no explanations or markdown."
    #         ),
    #     ]

    #     plotly_code = self.submit_prompt(message_log, **kwargs)
    #     # 确保代码不包含思考标签
    #     plotly_code = self._clean_thinking_tags(plotly_code)
    #     return self._sanitize_plotly_code(self._extract_python_code(plotly_code))

    def extract_sql(self, llm_response: str) -> str:
        """
        Example:
        ```python
        vn.extract_sql("Here's the SQL query in a code block: ```sql\nSELECT * FROM customers\n```")
        ```

        Extracts the SQL query from the LLM response. This is useful in case the LLM response contains other information besides the SQL query.
        Override this function if your LLM responses need custom extraction logic.

        Args:
            llm_response (str): The LLM response.

        Returns:
            str: The extracted SQL query.
        """

        # 1. 清理思考标签和特殊字符
        llm_response = self._clean_thinking_tags(llm_response)
        # If the llm_response contains a CTE (with clause), extract the last sql between WITH and ;
        sqls = re.findall(r"\bWITH\b .*?;", llm_response, re.DOTALL)
        if sqls:
            sql = sqls[-1]
            self.log(title="Extracted SQL", message=f"{sql}")
            return sql

        # If the llm_response is not markdown formatted, extract last sql by finding select and ; in the response
        sqls = re.findall(r"SELECT.*?;", llm_response, re.DOTALL)
        if sqls:
            sql = sqls[-1]
            self.log(title="Extracted SQL", message=f"{sql}")
            return sql

        # If the llm_response contains a markdown code block, with or without the sql tag, extract the last sql from it
        sqls = re.findall(r"```sql\n(.*)```", llm_response, re.DOTALL)
        if sqls:
            sql = sqls[-1]
            self.log(title="Extracted SQL", message=f"{sql}")
            return sql

        sqls = re.findall(r"```(.*)```", llm_response, re.DOTALL)
        if sqls:
            sql = sqls[-1]
            self.log(title="Extracted SQL", message=f"{sql}")
            return sql

        return llm_response

    # Removed the old implementation and the aggressive re.sub cleaner.
    # The old regexes like SELECT.*?; and WITH.*?; were problematic.
    # The final re.sub was removing valid characters like underscores and quotes.

    def generate_summary(self, question: str, df: pd.DataFrame, **kwargs) -> str:
        """
        **Example:**
        ```python
        vn.generate_summary("What are the top 10 customers by sales?", df)
        ```

        Generate a summary of the results of a SQL query, considering related documentation.

        Args:
            question (str): The question that was asked.
            df (pd.DataFrame): The results of the SQL query.
            **kwargs: Additional keyword arguments passed to underlying methods like get_related_documentation.

        Returns:
            str: The summary of the results of the SQL query.
        """
        # 获取相关文档
        doc_list = self.get_related_documentation(question, **kwargs)

        # 构建系统消息
        system_msg_base = f"You are a helpful data assistant. The user asked the question: '{question}'\n\n"

        # 添加文档上下文
        system_msg_with_docs = self.add_documentation_to_prompt(
            system_msg_base, doc_list, max_tokens=self.max_tokens
        )

        # 添加数据框信息
        final_system_msg = (
            system_msg_with_docs
            + f"""
            Note:For the Additional Context, if it is not relevant to the question, please ignore it.
            The following is a pandas DataFrame with the results of the query: 
            {df.to_markdown()}
            """
        )

        message_log = [
            self.system_message(
                final_system_msg
            ),  # 使用包含文档和数据框信息的最终系统消息
            # 确保分析准确，回答问题也依据数据，确保有所依据，不要回答和问题无关的内容
            self.user_message(
                """
                Based on the user's question and the provided context (including documentation if available), analyze the provided data (which you received in the system message). 
                Provide a summary that directly addresses the question. 
                If the question asks for an evaluation, assessment, or overall level (e.g., performance), include this analysis in your summary. 
                Ensure your entire response is accurate and strictly based on the provided data and context. 
                Be concise and focus on answering the query.

                Output constraints:
                - Do not output Python, matplotlib, plotly code, or any code block.
                - Do not provide runnable plotting scripts.
                - If visualization is relevant, only describe chart intent using existing frontend chart patterns
                  (bar, line, scatter, pie, histogram, indicator, table).
                - Refer users to the existing chart tab instead of suggesting local code execution.
                """
            ),
        ]

        summary = self.submit_prompt(message_log, need_reasoning=True, **kwargs)

        return summary

    def generate_summary_stream(self, question: str, df: pd.DataFrame, **kwargs):
        doc_list = self.get_related_documentation(question, **kwargs)

        system_msg_base = f"You are a helpful data assistant. The user asked the question: '{question}'\n\n"
        system_msg_with_docs = self.add_documentation_to_prompt(
            system_msg_base, doc_list, max_tokens=self.max_tokens
        )

        final_system_msg = (
            system_msg_with_docs
            + f"""
            Note:For the Additional Context, if it is not relevant to the question, please ignore it.
            The following is a pandas DataFrame with the results of the query: 
            {df.to_markdown()}
            """
        )

        message_log = [
            self.system_message(final_system_msg),
            self.user_message(
                """
                Based on the user's question and the provided context (including documentation if available), analyze the provided data (which you received in the system message). 
                Provide a summary that directly addresses the question. 
                If the question asks for an evaluation, assessment, or overall level (e.g., performance), include this analysis in your summary. 
                Ensure your entire response is accurate and strictly based on the provided data and context. 
                Be concise and focus on answering the query.

                Output constraints:
                - Do not output Python, matplotlib, plotly code, or any code block.
                - Do not provide runnable plotting scripts.
                - If visualization is relevant, only describe chart intent using existing frontend chart patterns
                  (bar, line, scatter, pie, histogram, indicator, table).
                - Refer users to the existing chart tab instead of suggesting local code execution.
                """
            ),
        ]

        if hasattr(self, "submit_prompt_stream"):
            yield from self.submit_prompt_stream(message_log, **kwargs)
            return

        summary = self.submit_prompt(message_log, need_reasoning=True, **kwargs)
        if isinstance(summary, dict):
            content = summary.get("content", "")
        else:
            content = summary or ""
        if content:
            yield content
