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
  console.log("Received document", kind)
  if (window.localStorage) {
    window.localStorage.setItem(kind, document);
  }

  let new_state = {...state};
  if (new_state.pending) {
    console.log("Document parked", kind)
    new_state.next = document;
  } else {
    console.log("Document submitted", kind)
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
    console.log("Rotating document", kind)
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


@rule
def query():
    return select_clause + opt(from_clause)


@rule
def select_clause():
    return SELECT + select_column_list


@rule(transparent=True)
def select_column_list():
    return alt(
        column_spec,
        select_column_list + COMMA + column_spec,
    )


@rule
def column_spec():
    return alt(
        STAR,
        expression + opt(alias),
    )


@rule
def alias():
    return AS + NAME


@rule
def from_clause():
    return FROM + table_list


@rule(transparent=True)
def table_list():
    return table_clause | (table_list + COMMA + table_clause)


@rule
def table_clause():
    return alt(
        table_expression + opt(alias),
        join_clause,
    )


@rule
def table_expression():
    return alt(
        NAME,
        LPAREN + query + RPAREN,
    )


@rule
def join_clause():
    return join_type + table_expression + ON + expression


@rule
def join_type():
    return (
        opt(
            alt(
                opt(alt(LEFT, RIGHT)) + OUTER,
                INNER,
                CROSS,
            )
        )
        + JOIN
    )


@rule
def expression():
    return NAME


BLANKS = Terminal("BLANKS", Re.set(" ", "\\t", "\\r", "\\n").plus())

# TODO: Case insensitivity? I don't know if I care- this grammar
#       tool is more about new languages than parsing existing ones,
#       and this SQL grammar is just a demo. Do people want case
#       ignoring lexers?
SELECT = Terminal("SELECT", "select")
AS = Terminal("AS", "as")
COMMA = Terminal("COMMA", ",")
STAR = Terminal("STAR", "*")
FROM = Terminal("FROM", "from")
WHERE = Terminal("WHERE", "where")
LPAREN = Terminal("LPAREN", "(")
RPAREN = Terminal("RPAREN", ")")

LEFT = Terminal("LEFT", "left")
RIGHT = Terminal("RIGHT", "right")
OUTER = Terminal("OUTER", "outer")
INNER = Terminal("INNER", "inner")
CROSS = Terminal("CROSS", "cross")
JOIN = Terminal("JOIN", "join")
ON = Terminal("ON", "on")

NAME = Terminal(
    "NAME",
    Re.seq(
        Re.set(("a", "z"), ("A", "Z"), "_"),
        Re.set(("a", "z"), ("A", "Z"), ("0", "9"), "_").star(),
    ),
)

SQL = Grammar(
    start=query,
    precedence=[],
    trivia=[BLANKS],
    name="SQL",
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


@rule
def File():
    # TODO: Make lists easier
    return _functions


@rule
def _functions():
    return Function | (_functions + Function)


@rule
def Function():
    return FN + NAME + ParamList + opt(ARROW + TypeExpr) + Block


@rule
def ParamList():
    return LPAREN + opt(_parameters) + RPAREN


@rule
def _parameters():
    # NOTE: The ungrammar in the reference does not talk about commas required between parameters
    #       so this massages it to make them required. Commas are in the list not the param, which
    #       is more awkward for processing but not terminally so.
    return (Param + opt(COMMA)) | (Param + COMMA + _parameters)


@rule
def Param():
    return NAME + COLON + TypeExpr


@rule
def TypeExpr():
    return NAME


@rule
def Block():
    return LCURLY + opt(_statements) + RCURLY


@rule
def _statements():
    return Stmt | _statements + Stmt


@rule
def Stmt():
    return StmtExpr | StmtLet | StmtReturn


@rule
def StmtExpr():
    return Expr + SEMICOLON


@rule
def StmtLet():
    return LET + NAME + EQUAL + Expr + SEMICOLON


@rule
def StmtReturn():
    return RETURN + Expr + SEMICOLON


@rule(error_name="expression")
def Expr():
    return ExprLiteral | ExprName | ExprParen | ExprBinary | ExprCall


@rule
def ExprLiteral():
    return INT | TRUE | FALSE


@rule
def ExprName():
    return NAME


@rule
def ExprParen():
    return LPAREN + Expr + RPAREN


@rule
def ExprBinary():
    return Expr + (PLUS | MINUS | STAR | SLASH) + Expr


@rule
def ExprCall():
    return Expr + ArgList


@rule
def ArgList():
    return LPAREN + opt(_arg_star) + RPAREN


@rule
def _arg_star():
    # Again, a deviation from the original. See _parameters.
    return (Expr + opt(COMMA)) | (Expr + COMMA + _arg_star)


BLANKS = Terminal("BLANKS", Re.set(" ", "\\t", "\\r", "\\n").plus())

TRUE = Terminal("TRUE", "true")
FALSE = Terminal("FALSE", "false")
INT = Terminal("INT", Re.set(("0", "9")).plus())
FN = Terminal("FN", "fn")
ARROW = Terminal("ARROW", "->")
COMMA = Terminal("COMMA", ",")
LPAREN = Terminal("LPAREN", "(")
RPAREN = Terminal("RPAREN", ")")
LCURLY = Terminal("LCURLY", "{")
RCURLY = Terminal("RCURLY", "}")
COLON = Terminal("COLON", ":")
SEMICOLON = Terminal("SEMICOLON", ";")
LET = Terminal("LET", "let")
EQUAL = Terminal("EQUAL", "=")
RETURN = Terminal("RETURN", "return")
PLUS = Terminal("PLUS", "+")
MINUS = Terminal("MINUS", "-")
STAR = Terminal("STAR", "*")
SLASH = Terminal("SLASH", "/")

NAME = Terminal(
    "NAME",
    Re.seq(
        Re.set(("a", "z"), ("A", "Z"), "_"),
        Re.set(("a", "z"), ("A", "Z"), ("0", "9"), "_").star(),
    ),
)

LGrammar = Grammar(
    name="L",
    start=File,
    trivia=[BLANKS],
    # Need a little bit of disambiguation for the symbol involved.
    precedence=[
        (Assoc.LEFT, [PLUS, MINUS]),
        (Assoc.LEFT, [STAR, SLASH]),
        (Assoc.LEFT, [LPAREN]),
    ],
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
