import os
import re
import sys
from typing import Union, Any, Dict, List, Optional
import pandas as pd
from tqdm import tqdm
from PIL import Image
import sqlite3

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sqlglot import parse_one, expressions
from sqlglot.executor import Table, execute
from sqlglot.planner import Plan, Join
from executor.visual_module import get_vqa_module


class NeuralDB(object):
    def __init__(self, db_path, check_same_thread=True):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path, check_same_thread=check_same_thread)

    def __str__(self):
        return str(self.execute_query("SELECT * FROM {}".format(self.table_name)))

    def execute_query(self, sql_query: str):
        out = self.conn.execute(sql_query)

        # Convert to list of dictionaries
        results = out.fetchall()
        unmerged_results = []

        headers = [d[0] for d in out.description]
        for i in range(len(results)):
            unmerged_results.append(results[i])

        return {"header": headers, "rows": unmerged_results}


class NeuralSQLExecutor:
    def __init__(
        self,
        mimic_iv_cxr_db_dir: str,
        mimic_cxr_image_dir: str,
        vqa_module_type: str = "yes",
        check_same_thread: bool = False,
    ) -> None:
        self.mimic_iv_cxr_db_dir = mimic_iv_cxr_db_dir
        self.mimic_cxr_image_dir = mimic_cxr_image_dir
        self.vqa_module_type = vqa_module_type

        self.db = NeuralDB(
            db_path=os.path.join(self.mimic_iv_cxr_db_dir, "mimic_iv_cxr.db"),
            check_same_thread=check_same_thread,
        )
        self.vqa_module = get_vqa_module(vqa_module_type)
        self.sid_to_iid_map = self._load_sid_to_iid_map()
        self.sid_to_ipath_map = self._load_sid_to_ipath_map()

    def _load_sid_to_iid_map(self) -> Dict[str, str]:
        """Load and map study IDs to image IDs from the database."""
        query = "SELECT DISTINCT study_id, image_id FROM TB_CXR"
        try:
            result = self.db.execute_query(query)
            if "study_id" not in result["header"] or "image_id" not in result["header"]:
                raise ValueError("Required columns are missing in the database result.")
            df = pd.DataFrame(result["rows"], columns=result["header"])
            if df["study_id"].nunique() != df["image_id"].nunique():
                raise ValueError("Study IDs and Image IDs are not uniquely paired.")
            sid_to_iid_map = {str(row["study_id"]): str(row["image_id"]) for _, row in df.iterrows()}
            return sid_to_iid_map
        except Exception as e:
            print(f"Failed to load the sid_to_iid_map: {e}")
            return {}

    def _load_sid_to_ipath_map(self) -> Dict[str, str]:
        """Load and map study IDs to image paths from the database."""
        query = "SELECT DISTINCT TB_CXR.subject_id, TB_CXR.study_id, TB_CXR.image_id FROM TB_CXR"
        try:
            result = self.db.execute_query(query)
            df = pd.DataFrame(result["rows"], columns=result["header"])
            if df["study_id"].nunique() != df["image_id"].nunique():
                raise ValueError("Study IDs and Image IDs are not uniquely paired.")
            sid_to_ipath_map = {}
            for _, row in df.iterrows():
                pid = str(row["subject_id"])
                sid = str(row["study_id"])
                iid = str(row["image_id"])
                ipath = os.path.join(self.mimic_cxr_image_dir, f"p{pid[:2]}/p{pid}/s{sid}/{iid}.jpg")
                if not os.path.exists(ipath):
                    print(f"Image file not found: {ipath}")
                    continue
                sid_to_ipath_map[sid] = ipath
            print(f"sid_to_ipath_map loaded. ({len(sid_to_ipath_map)} entries)")
            return sid_to_ipath_map
        except Exception as e:
            print(f"Failed to load the sid_to_ipath_map: {e}")
            return {}

    def _execute_vqa_function(self, image_batch: Union[str, List[str], List[Image.Image]], question_batch: Union[str, List[str]]) -> List[Any]:
        """Call the VQA API on a batch of images and questions and return the answers."""
        if isinstance(image_batch, str):
            image_batch = [Image.open(image_batch).convert("RGB")]
        elif isinstance(image_batch[0], str):
            image_batch = [Image.open(image_path).convert("RGB") for image_path in image_batch]
        elif isinstance(image_batch[0], Image.Image):
            pass
        else:
            raise ValueError("Invalid image_batch type")

        assert isinstance(image_batch[0], Image.Image)

        if isinstance(question_batch, str):
            question_batch = [question_batch]

        if len(image_batch) != len(question_batch):
            raise ValueError("The number of images and questions must be the same")

        answers = self.vqa_module(image_batch, question_batch)
        return answers

    def execute_sql(self, sql: str, use_table: bool = True, verbose: bool = False) -> Union[Table, Dict[str, Any]]:
        """Execute a SQL query."""
        if verbose:
            print(f"Executing SQL: {sql}")
        result = self.db.execute_query(sql)

        if result["header"] is None:
            target_header = parse_one(sql).args["expressions"][0].args["this"].sql()
            result["header"] = [target_header]

        if use_table:
            new_cols = tuple(col.lower() for col in result["header"])
            new_rows = [tuple(row) for row in result["rows"]]
            result = Table(columns=new_cols, rows=new_rows)

        return result

    def execute_vqa_function_in_nsql(self, nsql: str, tables: Dict[str, Table], image_batch_size: int = 32) -> Table:
        """Execute a VQA function within a SQL query."""
        # Parse the query and get the root node
        query = parse_one(nsql).sql().lower()
        root_node = parse_one(query)

        # Get the table name from the FROM clause
        table_name = root_node.args["from"].alias_or_name  # .upper()
        if table_name not in tables:
            raise ValueError(f"Table '{table_name}' not found in tables.")

        # Find all VQA function calls in the query
        parsed = parse_one(query)
        parsed_anonymous_list = [parsed_node for parsed_node in parsed.find_all(expressions.Anonymous) if re.search(r'FUNC_VQA\(".*?", .*?\)', parsed_node.sql())]

        parsed_results = {}
        for index, parsed_anonymous in enumerate(parsed_anonymous_list):
            question = parsed_anonymous.args["expressions"][0].sql()[1:-1]
            vqa_column = parsed_anonymous.sql().lower().strip()
            vqa_column_refined = vqa_column.replace("'s", "_s").replace('"', "'")  # NOTE: avoid sqlglot parser error
            parsed_results[index] = {
                "question": question,
                "api_col": vqa_column_refined,
                "output_key": vqa_column_refined,
            }
            query = query.replace(vqa_column, vqa_column_refined)

        for index, parsed_result in parsed_results.items():

            table = tables[table_name]

            question = parsed_result["question"]
            api_col = parsed_result["api_col"]
            output_key = parsed_result["output_key"]

            # Generate image batches for efficient VQA processing
            image_batches = []
            for idx in range(0, len(table.rows), image_batch_size):
                image_batch = []
                question_batch = []
                for row in table.rows[idx : idx + image_batch_size]:
                    study_id = str(row[table.columns.index("study_id")])
                    image_path = self.sid_to_ipath_map[study_id]
                    image_batch.append(image_path)
                    question_batch.append(question)
                image_batches.append((image_batch, question_batch))

            # Execute VQA on the image batches
            final_answer = []
            for image_batch, question_batch in tqdm(image_batches, desc="VQA inference"):
                answer_batch = self._execute_vqa_function(image_batch=image_batch, question_batch=question_batch)
                assert len(answer_batch) == len(image_batch), f"Answer count mismatch: {len(answer_batch)} != {len(image_batch)}"
                final_answer.extend(answer_batch)

            # Update the table with the VQA answers
            new_rows = []
            for row, answer in zip(table.rows, final_answer):
                if isinstance(answer, list):
                    for _answer in answer:
                        new_row = (*row, _answer)
                        new_rows.append(new_row)
                elif isinstance(answer, bool):
                    new_row = (*row, answer)
                    new_rows.append(new_row)
                else:
                    raise ValueError(f"Unexpected answer type: {type(answer)}")

            new_columns = table.columns + (output_key,)
            new_table = Table(columns=new_columns, rows=new_rows)

            # Update the tables dictionary with the new table
            tables[table_name] = new_table

            query = query.replace(api_col, f'"{api_col}"')

        # Execute the final query with the updated tables
        try:
            return execute(sql=query, tables=tables)
        except Exception:
            schema = {table_name: {col: "str" for col in table.columns} for table_name, table in tables.items()}
            return execute(sql=query, schema=schema, tables=tables)

    def execute_nsql(self, nsql: str, tables: Dict[str, Table] = None, verbose: bool = True) -> Union[Table, List[bool]]:
        """Execute a NeuralSQL query."""
        if tables is None:
            tables = {}

        # Normalize the SQL query
        nsql = parse_one(nsql).sql()

        # Parse the normalized query to get the root node
        query_root = parse_one(nsql)

        # Assert that the root node is one of the supported query types
        assert isinstance(
            query_root,
            (
                expressions.Select,
                expressions.Union,
                expressions.Except,
                expressions.Intersect,
            ),
        )

        # Check if the query contains VQA functions
        vqa_nodes = [node.this for node in query_root.walk() if isinstance(node, expressions.Anonymous) and node.this == "FUNC_VQA"]
        # assert vqa_nodes == ["FUNC_VQA"] or vqa_nodes == [], f"Unexpected VQA nodes: {vqa_nodes}"
        if not vqa_nodes and "FUNC_VQA" not in nsql:
            # If the query does not contain VQA functions, execute it as a typical query
            return self._execute_typical_query(nsql, tables)

        # Handle JOIN queries
        join_result = self._handle_join_query(query_root, nsql, tables)
        if join_result is not None:
            return join_result

        # Handle different types of query roots
        if isinstance(query_root, expressions.Select):
            return self._execute_select_query(query_root, nsql, tables)
        elif isinstance(query_root, (expressions.Union, expressions.Except, expressions.Intersect)):
            return self._execute_set_operation_query(query_root, nsql, tables)
        else:
            raise NotImplementedError(f"Unsupported query type: {type(query_root)}")

    def _handle_join_query(self, query_root: expressions.Select, nsql: str, tables: Dict[str, Table]) -> Optional[Union[Table, List[bool]]]:
        """Handle JOIN queries."""
        join_plan_list = [j for j in Plan(query_root).dag if type(j) == Join]
        if len(join_plan_list) == 1:
            simple_query = nsql
            tables = {}
            deps = join_plan_list[0].dependencies

            for dep in deps:
                join_query = f"""
                    SELECT {",".join([proj.sql() for proj in dep.projections])}
                    FROM {dep.source.sql()}
                    WHERE {dep.condition.sql()}
                """
                join_query = parse_one(join_query).sql()
                simple_query = simple_query.replace(f"({join_query}) AS ", "")

                if "FUNC_VQA" in join_query:
                    select_list = []
                    for select in parse_one(join_query).find_all(expressions.Select):
                        if "FUNC_VQA" in select.sql() and select.find_all(expressions.Anonymous):
                            select_list.append(select)
                    select = select_list[-1]
                    print(join_query)
                tables = {**tables, dep.name: self.execute_nsql(nsql=join_query)}

            return self.execute_nsql(nsql=simple_query, tables=tables)

        elif len(join_plan_list) > 1:
            raise NotImplementedError("Only support one join")
        else:
            return None

    def _execute_typical_query(self, nsql: str, tables: Dict[str, Table]) -> Union[Table, Dict[str, Any]]:
        """Execute a typical SQL query without VQA functions."""
        if tables:
            # Use sqlglot executor for queries with tables
            try:
                return execute(sql=nsql, tables=tables)
            except Exception:
                # If the query execution fails, try using a schema derived from the tables
                schema = {table_name: {col: "str" for col in table.columns} for table_name, table in tables.items()}
                return execute(sql=nsql, schema=schema, tables=tables)
        else:
            # Use sqlite3 executor for queries without tables
            return self.execute_sql(sql=nsql)

    def _execute_select_query(self, query_root: expressions.Select, nsql: str, tables: Dict[str, Table]) -> Union[Table, List[bool]]:
        """Execute a SELECT query."""
        if "from" not in query_root.args:
            return self._execute_subquery_operation(query_root, nsql)

        assert len(query_root.args["from"].unnest_operands()) == 1
        from_clause_node = query_root.args["from"].unnest_operands()[0]

        if isinstance(from_clause_node, expressions.Select):
            return self._execute_vqa_subquery(from_clause_node, query_root, nsql, tables)
        elif isinstance(from_clause_node, expressions.Table):
            return self._execute_vqa_table_query(query_root, nsql, tables)
        else:
            raise NotImplementedError(f"Unsupported FROM clause type: {type(from_clause_node)}")

    def _execute_subquery_operation(self, query_root: expressions.Select, nsql: str) -> List[bool]:
        """Execute a subquery operation (e.g., SELECT (SUBQUERY) AND (SUBQUERY))."""
        assert len(query_root.args["expressions"]) == 1
        if isinstance(query_root.args["expressions"][0], expressions.And):
            subquery_list = query_root.args["expressions"][0].unnest_operands()
            assert len(subquery_list) == 2
            answer1 = self.execute_nsql(subquery_list[0].sql())
            answer2 = self.execute_nsql(subquery_list[1].sql())
            if len(answer1) != len(answer2):
                raise ValueError("Expected answer count does not match")
            return [i[0] & j[0] for i, j in zip(answer1.rows, answer2.rows)]
        else:
            raise ValueError("Only support SELECT (SUBQUERY) AND (SUBQUERY)")

    def _execute_vqa_subquery(self, from_clause_node: expressions.Select, query_root: expressions.Select, nsql: str, tables: Dict[str, Table]) -> Table:
        """Execute a VQA subquery (e.g., SELECT FUNC_VQA() FROM (SUBQUERY))."""
        subquery = from_clause_node.sql()
        answer = self.execute_nsql(subquery)
        assert len(from_clause_node.expressions) <= 2
        table_alias = query_root.args["from"].alias_or_name
        tables[table_alias] = answer
        nsql = nsql.replace(query_root.args["from"].sql(), f"FROM {table_alias}")
        return self.execute_vqa_function_in_nsql(nsql, tables)

    def _execute_vqa_table_query(self, query_root: expressions.Select, nsql: str, tables: Dict[str, Table]) -> Union[Table, List[bool]]:
        """Execute a VQA table query (e.g., SELECT FUNC_VQA() FROM TB WHERE ...)."""
        select_col_clause_node = query_root.args["expressions"][0]

        if isinstance(select_col_clause_node, expressions.Anonymous):
            return self._execute_anonymous_vqa_query(select_col_clause_node, nsql, tables)
        elif isinstance(select_col_clause_node, (expressions.And, expressions.Or)):
            return self._execute_logical_vqa_query(select_col_clause_node, nsql, tables)
        elif isinstance(select_col_clause_node, expressions.Paren):
            return self._execute_paren_vqa_query(select_col_clause_node, nsql)
        elif isinstance(select_col_clause_node, expressions.Column):
            return self._execute_column_vqa_query(query_root, nsql)
        else:
            raise NotImplementedError(f"Unsupported select column type: {type(select_col_clause_node)}")

    def _execute_anonymous_vqa_query(self, select_col_clause_node: expressions.Anonymous, nsql: str, tables: Dict[str, Table]) -> Table:
        """Execute an anonymous VQA query (e.g., SELECT FUNC_VQA() FROM TB_CXR WHERE ...)."""
        select_col_node = select_col_clause_node.args["expressions"][1]
        select_col_clause = select_col_clause_node.sql()
        assert select_col_node.sql() == "tb_cxr.study_id"
        subquery = nsql.replace(select_col_clause, select_col_node.sql())
        answer = self.execute_sql(sql=subquery)
        table_alias = "tb_cxr"
        tables[table_alias] = answer
        nsql = f"select {select_col_clause} from {table_alias}"
        return self.execute_vqa_function_in_nsql(nsql, tables)

    def _execute_logical_vqa_query(self, select_col_clause_node: Union[expressions.And, expressions.Or], nsql: str, tables: Dict[str, Table]) -> Table:
        """Execute a logical VQA query (e.g., SELECT (FUNC_VQA() OP FUNC_VQA()) FROM TB_CXR WHERE ...)."""
        subquery = nsql
        select_col_wo_op_clause_node = select_col_clause_node.unnest_operands()
        for select_col_clause_node in select_col_wo_op_clause_node:
            select_col_node = select_col_clause_node.args["expressions"][1]
            assert select_col_node.sql() == "tb_cxr.study_id"
        subquery = subquery.replace(select_col_clause_node.sql(), select_col_node.sql())
        answer = self.execute_sql(sql=subquery)
        table_alias = "tb_cxr"
        tables[table_alias] = answer
        nsql = f"select {select_col_clause_node.sql()} from {table_alias}"
        return self.execute_vqa_function_in_nsql(nsql, tables)

    def _execute_paren_vqa_query(self, select_col_clause_node: expressions.Paren, nsql: str) -> List[bool]:
        """Execute a parenthesized VQA query (e.g., SELECT (FUNC_VQA() = TRUE/FALSE) FROM TB_CXR WHERE ...)."""
        if isinstance(select_col_clause_node.unnest(), expressions.EQ):
            _select_col_clause_node, bool_val = select_col_clause_node.unnest().unnest_operands()
            subquery = nsql.replace(select_col_clause_node.sql(), _select_col_clause_node.sql())
            answer = self.execute_nsql(subquery)

            # Apply answer + bool_val
            bool_val = bool_val.sql()
            assert isinstance(answer, Table)
            new_rows = [(r[0] == bool_val,) for r in answer.rows]
            new_columns = answer.columns
            return Table(columns=new_columns, rows=new_rows)
        else:
            raise NotImplementedError()

    def _execute_column_vqa_query(self, query_root: expressions.Select, nsql: str) -> Table:
        """Execute a column VQA query (e.g., SELECT COLUMN(S) FROM TB_CXR WHERE TB_CXR.STUDY_ID IN (SUBQUERY:FUNC_VQA()))."""
        select_list = [select.sql() for select in query_root.find_all(expressions.Select) if "FUNC_VQA" in select.sql() and select.find_all(expressions.Anonymous)]
        assert len(select_list) == 2, select_list
        answer = self.execute_nsql(select_list[-1])
        in_items = ",".join([str(row[len(answer.columns) - 1]) for row in answer.rows])
        nsql = nsql.replace(select_list[-1], in_items)
        if not in_items:
            select_col_clause_of_root_node = query_root.args["expressions"]
            return Table(columns=[sel_col_node.name for sel_col_node in select_col_clause_of_root_node], rows=[])
        else:
            return self.execute_nsql(nsql)

    def _execute_set_operation_query(self, query_root: Union[expressions.Union, expressions.Except, expressions.Intersect], nsql: str, tables: Dict[str, Table]) -> Table:
        """Execute a set operation query (e.g., SELECT (FUNC_VQA() ...) FROM ... SET_OP SELECT (FUNC_VQA() ...) FROM ...)."""
        select_clause_nodes = parse_one(nsql).unnest_operands()
        assert len(select_clause_nodes) == 2
        sql_components = []
        for idx, select_clause_node in enumerate(select_clause_nodes):
            query = select_clause_node.sql()
            answer = self.execute_nsql(nsql=query)
            assert len(answer.columns) == 1
            tables[f"t{idx}"] = answer
            new_col = select_clause_node.expressions[0].sql().lower().replace('"', "'")
            sql_components.append(f'select "{new_col}" from t{idx}')

        operator = {expressions.Union: "union", expressions.Except: "except", expressions.Intersect: "intersect"}[type(query_root)]
        sql = f" {operator} ".join(sql_components)
        return self.execute_nsql(sql, tables)
