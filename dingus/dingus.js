// TODO: Abstract/Factor
let pending_grammar = null;
let next_grammar = null;

let pending_document = null;
let next_document = null;

const STATUS = document.getElementById("status-line");

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

function submit_document(code) {
  if (pending_document) {
    console.log("Document still pending, parking it");
    next_document = code;
  } else {
    pending_document = code;
    worker.postMessage({kind: "document", data: code});
    console.log("Document posted");
  }
}

const worker = new Worker('worker.js');
worker.onmessage = (e) => {
  const message = e.data;
  if (message.kind === "grammar_status") {
    STATUS.innerText = message.message;

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

  if (message.kind === "doc_status") {
    STATUS.innerText = message.message;

    if ((message.status === "ok") || (message.status === "error")) {
      pending_document = null;
      if (next_document) {
        pending_document = next_document;
        next_document = null;

        worker.postMessage({kind: "document", data: pending_document});
        console.log("Posted another document");
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

  let grammar_change_timer_id = null;
  grammar_editor.doc.on("change", () => {
    clearTimeout(grammar_change_timer_id);
    grammar_change_timer_id = setTimeout(() => {
      grammar_change_timer_id = null;
      submit_grammar(grammar_editor.doc.getValue());
    }, 100);
  });

  const input_editor = CodeMirror.fromTextArea(
    document.getElementById("input"),
    {
      lineNumbers: true,
    },
  );

  let input_change_timer_id = null;
  input_editor.doc.on("change", () => {
    clearTimeout(input_change_timer_id);
    input_change_timer_id = setTimeout(() => {
      input_change_timer_id = null;
      submit_document(input_editor.doc.getValue());
    }, 100);
  });
}

setup_editors();
