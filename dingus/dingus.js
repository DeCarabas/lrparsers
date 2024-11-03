const STATUS = document.getElementById("status-line");
const OUTPUT = document.getElementById("output-root");
const TREE_BUTTON = document.getElementById("tree-button");
const ERRORS_BUTTON = document.getElementById("errors-button");

const EXAMPLE_SELECT = document.getElementById("example-select");

const INITIAL_STATE = {
  worker: null,
  status: "Initializing...",
  output_mode: "errors",
  tree: null,
  errors: [],
  grammar: {
    last: null,
    pending: null,
    next: null,
  },
  input: {
    last: null,
    pending: null,
    next: null,
  },
};


/*
 * Render a tree `node` into the DOM.
 */
function render_tree_node(node, input_editor) {
  const tree_div = document.createElement("div");
  tree_div.classList.add("parsed-node");

  const node_label = document.createElement("a");
  node_label.innerText = node.name;
  node_label.setAttribute("href", "#");
  node_label.onclick = () => {
    const doc = input_editor.doc;
    doc.setSelection(
      doc.posFromIndex(node.start),
      doc.posFromIndex(node.end),
      {scroll: true},
    );
  };
  if (node.start == node.end) {
    node_label.classList.add("parsed-error-node");
  }
  tree_div.appendChild(node_label);


  if (node.kind === "tree") {
    tree_div.classList.add("parsed-tree");
    for (const child of node.children) {
      tree_div.appendChild(render_tree_node(child, input_editor));
    }
  } else {
    tree_div.classList.add("parsed-token");
  }

  return tree_div;
}


/*
 * Render the state out into the DOM, the bindings to which we have already
 * established. Fully side-effecting, no shadow DOM here.
 */
function render_state(state, input_editor) {
  STATUS.innerText = state.status;

  const error_count = state.errors.length;
  if (error_count > 0) {
    ERRORS_BUTTON.disabled = false;
    if (error_count > 1) {
      ERRORS_BUTTON.innerText = `${error_count} errors`;
    } else {
      ERRORS_BUTTON.innerText = `1 error`;
    }
  } else {
    ERRORS_BUTTON.innerText = "No errors";
    ERRORS_BUTTON.disabled = true;
  }

  if (state.tree) {
    TREE_BUTTON.innerText = "Tree";
    TREE_BUTTON.disabled = false;
  } else {
    TREE_BUTTON.innerText = "No tree";
    TREE_BUTTON.disabled = true;
  }

  if (state.output_mode === "errors") {
    const error_node = document.createElement("div");
    error_node.classList.add("error-panel");
    if (state.errors.length == 0) {
      if (state.tree) {
        error_node.innerText = "No errors. Click the 'tree' button to see the parse tree.";
      } else {
        error_node.innerText = "No errors.";
      }
    } else {
      const ul = document.createElement("ul");
      ul.replaceChildren(...state.errors.map(e => {
        const li = document.createElement("li");
        li.innerText = e;
        return li;
      }));
      error_node.appendChild(ul);
    }

    OUTPUT.replaceChildren(error_node);
  } else if (state.output_mode === "tree") {
    if (state.tree) {
      OUTPUT.replaceChildren(render_tree_node(state.tree, input_editor));
    } else {
      if (state.errors.length === 1) {
        OUTPUT.replaceChildren("No parse tree. Click the 'error' button to see the error.");
      } else if (state.errors.length > 0) {
        OUTPUT.replaceChildren("No parse tree. Click the 'error' button to see the errors.");
      } else {
        OUTPUT.replaceChildren("No parse tree.");
      }
    }
  }
}

/*
 * Post a changed document out to the worker, if the worker can take it,
 * otherwise just queue it for submission.
 */
function post_document(worker, kind, state, document) {
  if (window.localStorage) {
    window.localStorage.setItem(kind, document);
  }

  let new_state = {...state};
  if (new_state.pending) {
    new_state.next = document;
  } else {
    new_state.pending = document;
    new_state.next = null;
    worker.postMessage({kind, data: document});
  }
  return new_state;
}

/*
 * Handle a document submission by rotating in the next document to be
 * submitted. (Documents flow from next -> pending -> last.)
 */
function rotate_document(worker, kind, state) {
  let new_state = {...state, last: state.pending, pending: null};
  if (new_state.next) {
    new_state.pending = new_state.next;
    new_state.next = null;
    worker.postMessage({kind, data: new_state.pending});
  }
  return new_state;
}

/*
 * Update the state given the message and return a new state.
 *
 * This can be side-effecting, in that it might also post messages to the worker
 * (to rotate documents and whatnot). (Maybe we should do something about that?)
 */
function update(state, message) {
  let new_state = {...state};
  if (message.message) {
    new_state.status = message.message;
  }

  if (message.kind === "grammar") {
    if (message.status === "changed") {
      new_state.grammar = post_document(
        new_state.worker,
        "grammar",
        new_state.grammar,
        message.data,
      );
    } else if (message.status === "ok" || message.status === "error") {
      new_state.grammar = rotate_document(new_state.worker, "grammar", new_state.grammar);

      if (message.status === "ok") {
        // Re-submit the input, using whatever the latest document state was.
        new_state.input = post_document(
          new_state.worker,
          "input",
          new_state.input,
          new_state.input.next || new_state.input.pending || new_state.input.last,
        );
      }

      if (message.status === "error") {
        new_state.errors = message.errors;
        new_state.tree = null;
      }
    }
  }

  if (message.kind === "input") {
    if (message.status === "changed") {
      new_state.input = post_document(
        new_state.worker,
        "input",
        new_state.input,
        message.data,
      );
    } else if (message.status === "ok" || message.status === "error") {
      new_state.input = rotate_document(new_state.worker, "input", new_state.input);

      if (message.status === "ok") {
        // On parse, there can still be errors even if the status is ok.
        new_state.tree = message.tree;
        new_state.errors = message.errors;
      }

      if (message.status === "error") {
        new_state.tree = null;
        new_state.errors = message.errors;
      }
    }
  }

  if (message.kind === "tree_button") {
    new_state.output_mode = "tree";
  }

  if (message.kind === "errors_button") {
    new_state.output_mode = "errors";
  }

  //console.log(state, message, new_state);
  return new_state;
}

/*
 * Construct a new "message handler" by wrapping up the state into an update/
 * render loop.
 */
function message_handler(worker, input_editor) {
  let state = {...INITIAL_STATE, worker};
  render_state(state, input_editor);

  return (message) => {
    state = update(state, message);
    render_state(state, input_editor);
  };
}

// And we can set up the codemirror editors to publish their changes.
function setup_editors() {
  function setup_editor(kind, editor, handler) {
    let change_timer_id = null;
    editor.doc.on("change", () => {
      clearTimeout(change_timer_id);
      change_timer_id = setTimeout(() => {
        change_timer_id = null;
        handler({
          kind,
          status: "changed",
          data: editor.doc.getValue(),
        });
      }, 100);
    });

    if (window.localStorage) {
      const value = window.localStorage.getItem(kind);
      if (value) {
        editor.doc.setValue(value);
      }
    }
  }

  // Construct the editors and worker and stuff but don't wire the handler
  // yet...
  const worker = new Worker('worker.js');
  const grammar_editor = CodeMirror.fromTextArea(
    document.getElementById("grammar"),
    {
      lineNumbers: true,
      mode: "python",
    },
  );
  const input_editor = CodeMirror.fromTextArea(
    document.getElementById("input"),
    {
      lineNumbers: true,
    },
  );

  // ...now we can construct the handler with what it needs....
  const handler = message_handler(worker, input_editor);

  // ...and finally hook the event sources to the handler.
  worker.onmessage = (e) => {
    const message = e.data;
    handler(message);
  };

  setup_editor("grammar", grammar_editor, handler);
  setup_editor("input", input_editor, handler);

  TREE_BUTTON.onclick = () => handler({kind: "tree_button"});
  ERRORS_BUTTON.onclick = () => handler({kind: "errors_button"});

  for (const example of EXAMPLES) {
    const opt = document.createElement("option");
    opt.value = example.id;
    opt.text = example.name;
    EXAMPLE_SELECT.add(opt, null);
  }

  EXAMPLE_SELECT.onchange = () => {
    if (EXAMPLE_SELECT.selectedIndex > 0) {
      const index = EXAMPLE_SELECT.selectedIndex - 1;
      const example = EXAMPLES[index];

      if (window.confirm("If you continue, any changes in the window will be lost. OK?")) {
        grammar_editor.doc.setValue(example.grammar.trim());
        input_editor.doc.setValue(example.input.trim());
      }

      EXAMPLE_SELECT.selectedIndex = 0;
    }
  };
}

const EXAMPLES = [
  {
    "id": "sql",
    "name": "SQL (well, sorta)",
    "grammar": `
# A silly little SQL grammar. Incomplete, but you get it, right?
from parser import *

class SQLGrammar(Grammar):
    start = "query"
    trivia = ["BLANKS"]

    precedence = [
    ]

    @rule
    def query(self):
        return self.select_clause + opt(self.from_clause)

    @rule
    def select_clause(self):
        return self.SELECT + self.select_column_list

    @rule(transparent=True)
    def select_column_list(self):
        return alt(
          self.column_spec,
          self.select_column_list + self.COMMA + self.column_spec,
        )

    @rule
    def column_spec(self):
        return alt(
          self.STAR,
          self.expression + opt(self.alias),
        )

    @rule
    def alias(self):
      return self.AS + self.NAME

    @rule
    def from_clause(self):
      return self.FROM + self.table_list

    @rule(transparent=True)
    def table_list(self):
      return (
        self.table_clause |
        (self.table_list + self.COMMA + self.table_clause)
      )

    @rule
    def table_clause(self):
      return alt(
        self.table_expression + opt(self.alias),
        self.join_clause,
      )

    @rule
    def table_expression(self):
      return alt(
        self.NAME,
        self.LPAREN + self.query + self.RPAREN,
      )

    @rule
    def join_clause(self):
      return self.join_type + self.table_expression + self.ON + self.expression

    @rule
    def join_type(self):
      return opt(alt(
        opt(alt(self.LEFT, self.RIGHT)) + self.OUTER,
        self.INNER,
        self.CROSS,
      )) + self.JOIN

    @rule
    def expression(self):
        return self.NAME

    BLANKS = Terminal(Re.set(" ", "\t", "\r", "\n").plus())

    # TODO: Case insensitivity? I don't know if I care- this grammar
    #       tool is more about new languages than parsing existing ones,
    #       and this SQL grammar is just a demo. Do people want case
    #       ignoring lexers?
    SELECT = Terminal("select")
    AS = Terminal("as")
    COMMA = Terminal(",")
    STAR = Terminal("*")
    FROM = Terminal("from")
    WHERE = Terminal("where")
    LPAREN = Terminal("(")
    RPAREN = Terminal(")")

    LEFT = Terminal("left")
    RIGHT = Terminal("right")
    OUTER = Terminal("outer")
    INNER = Terminal("inner")
    CROSS = Terminal("cross")
    JOIN = Terminal("join")
    ON = Terminal("on")

    NAME = Terminal(
        Re.seq(
            Re.set(("a", "z"), ("A", "Z"), "_"),
            Re.set(("a", "z"), ("A", "Z"), ("0", "9"), "_").star(),
        ),
    )
`,
    "input": `
select
    *,
    mumble as grumble,
    humble
from
    bumble as stumble,
    (select asdf) as y,
    left outer join foo on asdf
`
  },
  {
    "id": "matklad",
    "name": "L (a resilient parsing example)",
    "grammar": `
# A grammar based on
# https://matklad.github.io/2023/05/21/resilient-ll-parsing-tutorial.html
from parser import *

class LGrammar(Grammar):
    start = "File"
    trivia = ["BLANKS"]

    # Need a little bit of disambiguation for the symbol involved.
    precedence = [
        (Assoc.LEFT, ["PLUS", "MINUS"]),
        (Assoc.LEFT, ["STAR", "SLASH"]),
        (Assoc.LEFT, ["LPAREN"]),
    ]

    @rule
    def File(self):
        # TODO: Make lists easier
        return self._functions

    @rule
    def _functions(self):
        return self.Function | (self._functions + self.Function)

    @rule
    def Function(self):
        return self.FN + self.NAME + self.ParamList + opt(self.ARROW + self.TypeExpr) + self.Block

    @rule
    def ParamList(self):
        return self.LPAREN + opt(self._parameters) + self.RPAREN

    @rule
    def _parameters(self):
        # NOTE: The ungrammar in the reference does not talk about commas required between parameters
        #       so this massages it to make them required. Commas are in the list not the param, which
        #       is more awkward for processing but not terminally so.
        return (self.Param + opt(self.COMMA)) | (self.Param + self.COMMA + self._parameters)

    @rule
    def Param(self):
        return self.NAME + self.COLON + self.TypeExpr

    @rule
    def TypeExpr(self):
        return self.NAME

    @rule
    def Block(self):
        return self.LCURLY + opt(self._statements) + self.RCURLY

    @rule
    def _statements(self):
        return self.Stmt | self._statements + self.Stmt

    @rule
    def Stmt(self):
        return self.StmtExpr | self.StmtLet | self.StmtReturn

    @rule
    def StmtExpr(self):
        return self.Expr + self.SEMICOLON

    @rule
    def StmtLet(self):
        return self.LET + self.NAME + self.EQUAL + self.Expr + self.SEMICOLON

    @rule
    def StmtReturn(self):
        return self.RETURN + self.Expr + self.SEMICOLON

    @rule(error_name="expression")
    def Expr(self):
        return self.ExprLiteral | self.ExprName | self.ExprParen | self.ExprBinary | self.ExprCall

    @rule
    def ExprLiteral(self):
        return self.INT | self.TRUE | self.FALSE

    @rule
    def ExprName(self):
        return self.NAME

    @rule
    def ExprParen(self):
        return self.LPAREN + self.Expr + self.RPAREN

    @rule
    def ExprBinary(self):
        return self.Expr + (self.PLUS | self.MINUS | self.STAR | self.SLASH) + self.Expr

    @rule
    def ExprCall(self):
        return self.Expr + self.ArgList

    @rule
    def ArgList(self):
        return self.LPAREN + opt(self._arg_star) + self.RPAREN

    @rule
    def _arg_star(self):
        # Again, a deviation from the original. See _parameters.
        return (self.Expr + opt(self.COMMA)) | (self.Expr + self.COMMA + self._arg_star)

    BLANKS = Terminal(Re.set(" ", "\\t", "\\r", "\\n").plus())

    TRUE = Terminal("true")
    FALSE = Terminal("false")
    INT = Terminal(Re.set(("0", "9")).plus())
    FN = Terminal("fn")
    ARROW = Terminal("->")
    COMMA = Terminal(",")
    LPAREN = Terminal("(")
    RPAREN = Terminal(")")
    LCURLY = Terminal("{")
    RCURLY = Terminal("}")
    COLON = Terminal(":")
    SEMICOLON = Terminal(";")
    LET = Terminal("let")
    EQUAL = Terminal("=")
    RETURN = Terminal("return")
    PLUS = Terminal("+")
    MINUS = Terminal("-")
    STAR = Terminal("*")
    SLASH = Terminal("/")

    NAME = Terminal(
        Re.seq(
            Re.set(("a", "z"), ("A", "Z"), "_"),
            Re.set(("a", "z"), ("A", "Z"), ("0", "9"), "_").star(),
        ),
    )
`,
    "input": `
fn dingus(x:f64, y:f64) -> f64 {
  return 23;
}

fn what() {
}

fn something() {
  return 123 * 12;
}
`,
  },
]

setup_editors();
