let pending_grammar = null;
let next_grammar = null;

function submit_grammar(code) {
  if (pending_grammar) {
    console.log("Grammar still pending, parking it");
    next_grammar = code;
  } else {
    pending_grammar = code;
    worker.postMessage({kind: "grammar_module", data: code});
    console.log("Grammar posted");
  }
}

const worker = new Worker('worker.js');
worker.onmessage = (e) => {
  const message = e.data;
  if (message.kind === "grammar_status") {
    if ((message.status === "ok") || (message.status === "error")) {
      pending_grammar = null;
      if (next_grammar) {
        pending_grammar = next_grammar;
        next_grammar = null;

        worker.postMessage({kind: "grammar_module", data: pending_grammar});
        console.log("Posted another grammar");
      }
    }
  }
};


function setup_editors() {
  const grammar_editor = CodeMirror.fromTextArea(
    document.getElementById("grammar"),
    {
      lineNumbers: true,
      mode: "python",
      value: "from parser import Grammar\n\nclass MyGrammar(Grammar):\n    pass\n",
    },
  );

  let change_timer_id = null;
  grammar_editor.doc.on("change", () => {
    clearTimeout(change_timer_id);
    change_timer_id = setTimeout(() => {
      change_timer_id = null;
      submit_grammar(grammar_editor.doc.getValue());
    }, 100);
  });

  const input_editor = CodeMirror.fromTextArea(
    document.getElementById("input"),
    {
      lineNumbers: true,
    },
  );
}

setup_editors();
