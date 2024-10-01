const PARSER_PACKAGE = "./wheel/lrparsers-0.7.9-py3-none-any.whl"


// Load the whole pyodide thingy.
importScripts("pyodide/pyodide.js");

const dingus_module = {
  post_grammar_status: function (message) {
    console.log("Grammar Status:", message);
    postMessage({kind: "grammar_status", status: "loading", message});
  },

  post_grammar_loaded: function (name) {
    const msg = "Grammar '" + name + "' loaded";
    console.log(msg);
    postMessage({kind: "grammar_status", status: "ok", message: msg});
  },

  post_grammar_error: function(error) {
    console.log("Grammar Error:", error);
    postMessage({kind:"grammar_status", status: "error", message: error});
  },
};

async function setup_python() {
  dingus_module.post_grammar_status("Loading python....");
  const pyodide = await loadPyodide({
    packages: ["micropip"],
  });
  pyodide.setStdout({ batched: (msg) => console.log(msg) }); // TODO: I know this is an option above.

  // TODO: Do I actually want micropip? Probably not?
  dingus_module.post_grammar_status("Installing parser package...");
  const micropip = pyodide.pyimport("micropip");
  await micropip.install(PARSER_PACKAGE);

  dingus_module.post_grammar_status("Configuring dingus...");
  pyodide.registerJsModule("dingus", dingus_module);

  pyodide.runPython(`
import dingus
import parser
import pyodide.code

GRAMMAR_GLOBALS = {}
GRAMMAR = None
PARSE_TABLE = None
LEXER = None

def eval_grammar(code):
  global GRAMMAR_GLOBALS
  global GRAMMAR
  global PARSE_TABLE
  global LEXER

  try:
    dingus.post_grammar_status("Evaluating grammar...")
    pyodide.code.eval_code(code, globals=GRAMMAR_GLOBALS)

    grammar = None
    for key, value in GRAMMAR_GLOBALS.items():
      if isinstance(value, type) and issubclass(value, parser.Grammar) and value is not parser.Grammar:
        value = value()

      if isinstance(value, parser.Grammar):
        if grammar is None:
          grammar = value
        else:
          raise Exception("More than one Grammar found in the file")

    if grammar is None:
      raise Exception("No grammar definition, define or instantiate a class that inherits from parser.Grammar")

    GRAMMAR = grammar

    dingus.post_grammar_status("Building parse table...")
    PARSE_TABLE = grammar.build_table()

    dingus.post_grammar_status("Building lexer...")
    LEXER = grammar.compile_lexer()

    dingus.post_grammar_loaded(grammar.name)
  except Exception as e:
    dingus.post_grammar_error(f"{e}")
`);

  dingus_module.post_grammar_status("Ready.");
  self.pyodide = pyodide;
  return pyodide;
}
const pyodide_promise = setup_python();


async function load_grammar_module(code) {
  const pyodide = self.pyodide;

  // console.log("Running...");
  const my_fn = pyodide.globals.get("eval_grammar");
  my_fn(code);
  my_fn.destroy();
}

self.onmessage = async function(event) {
  await pyodide_promise;

  try {
    const { kind, data } = event.data;
    if (kind === "grammar_module") {
      try {
        await load_grammar_module(data);
      } catch (e) {
        console.log("INTERNAL ERROR:", e.message);
        postMessage({error: e.message});
      }
    }
  } catch (wtf) {
    console.log("WTF?");
    console.log(wtf);
  }
};
