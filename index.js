//import * as tf from '@tensorflow/tfjs';

console.log(tf);
window.tf = tf;

function outputText(text) {
  const textDiv = document.createElement('div');
  textDiv.innerText = text;
  document.querySelector('#output').append(textDiv);
}

function clearOutputText() {
  document.querySelector('#output').innerText = '';
}

async function runDeferred(f) {
  return new Promise(function(resolve, reject) {
    setTimeout(async () => {
      resolve(await f());
    }, 10);
  });
}

let timerStart = 0;
let timerText = '';
function startTimer(text) {
  timerStart = Date.now();
  timerText = text;
  outputText(text);
}

function endTimer() {
  const timeElapsed = Date.now() - timerStart;
  outputText('finished after ' + timeElapsed + ' ms: ' + timerText);
}

async function loadTransformerLayersModel() {
  startTimer('loading transformer graph model');
  //window.model = await tf.loadLayersModel('/keras_transformer_functional_tfjs/model.json');
  window.encoder = await tf.loadGraphModel('./pt_en_tensorflow_model_tfjs_graph_encoder/model.json');
  window.decoder = await tf.loadGraphModel('./pt_en_tensorflow_model_tfjs_graph_decoder/model.json');
  endTimer();
}

function make_tokenizer_funcs(tokenizer_en_def) {
  var token_lengths = new Set()
  var token_lengths_list = []
  for (var x of Object.keys(tokenizer_en_def.model.vocab)) {
    var token_length = x.length;
    if (token_lengths.has(token_length)) {
      continue
    }
    token_lengths_list.push(token_length)
    token_lengths.add(token_length)
  }
  token_lengths_list.sort((x,y) => y - x)
  
  function text_to_ids(input_text) {
    var output = []
    var cur_pos = 0;
    while (cur_pos < input_text.length) {
      for (var token_length of token_lengths_list) {
        if (token_length > input_text.length - cur_pos) {
          continue
        }
        var cur_text = input_text.substr(cur_pos, token_length)
        if (tokenizer_en_def.model.vocab[cur_text] != undefined) {
          output.push(tokenizer_en_def.model.vocab[cur_text])
          cur_pos += token_length
          break
        }
        //console.log(cur_text)
      }
    }
    return output
  }
  
  var id_to_vocab = []
  for (var [k, v] of Object.entries(tokenizer_en_def.model.vocab)) {
    id_to_vocab[v] = k
  }
  
  function ids_to_text(token_id_list) {
    var output = []
    for (var x of token_id_list) {
      output.push(id_to_vocab[x])
    }
    return output.join('');
  }

  var vocab_size = Object.keys(tokenizer_en_def.model.vocab).length;

  return {
    'text_to_ids': text_to_ids,
    'ids_to_text': ids_to_text,
    'vocab_size': vocab_size,
  }
}

window.tok_en = make_tokenizer_funcs(tokenizer_en)
window.tok_pt = make_tokenizer_funcs(tokenizer_pt)

function create_padding_mask(seq) {
  var seq2 = tf.cast(tf.equal(seq, 0), 'float32');
  return seq2.expandDims(1).expandDims(1);
}

function create_look_ahead_mask(size) {
  var mask = tf.sub(1, tf.linalg.bandPart(tf.ones([size, size]), -1, 0));
  return mask
}

function create_combined_mask(tar) {
  var look_ahead_mask = create_look_ahead_mask(tar.shape[1]);
  var dec_target_padding_mask = create_padding_mask(tar);
  var combined_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask);
  return combined_mask;
}

async function precompute_encoder_output(inp_sentence) {
  var start_token = [tok_pt.vocab_size];
  var end_token = [tok_pt.vocab_size + 1];
  var token_ids = tok_pt.text_to_ids(inp_sentence);
  var inp_sentence = start_token.concat(token_ids).concat(end_token);
  inp_sentence = tf.tensor(inp_sentence, undefined, 'int32')
  var encoder_input = tf.expandDims(inp_sentence, 0);
  var enc_padding_mask = create_padding_mask(encoder_input);
  //var enc_output = encoder([encoder_input, tf.constant(False, dtype=tf.bool), enc_padding_mask])
  var enc_output = await encoder.executeAsync({
    'input_1:0': encoder_input,// tf.tensor(encoder_input, undefined, 'int32'),
    'input_2:0': tf.tensor(false, [], 'bool'),
    //'input_3:0': training_tensor, //tf.tensor([false]),
    //training_tensor,
    //tf.tensor(inputs[2]),
    'input_3:0': enc_padding_mask, //tf.tensor(enc_padding_mask),
    //tf.tensor(tmp, [40, 1, 1, 40]),
  });
  return enc_output;
}

var MAX_LENGTH = 40;

async function evaluate(inp_sentence, prefix, enc_output) {

  var start_token = [tok_pt.vocab_size];
  var end_token = [tok_pt.vocab_size + 1];
  var token_ids = tok_pt.text_to_ids(inp_sentence);
  var inp_sentence = start_token.concat(token_ids).concat(end_token);
  inp_sentence = tf.tensor(inp_sentence, undefined, 'int32')
  var encoder_input = tf.expandDims(inp_sentence, 0);
  var decoder_input = tf.tensor([tok_en.vocab_size], undefined, 'int32');
  var output = tf.expandDims(decoder_input, 0);
  var dec_padding_mask = create_padding_mask(encoder_input);
  var prefix_ids = [];
  var prefix_tokens = tok_en.text_to_ids(prefix);
  for (var token_id of prefix_tokens) {
    prefix_ids.push(token_id);
  }
  for (var i = 0; i < MAX_LENGTH; ++i) {
    var combined_mask = create_combined_mask(output);
    // decoder([output, enc_output, tf.constant(False, dtype=tf.bool), combined_mask, dec_padding_mask])
    var predictions = await decoder.executeAsync({
      'input_1:0': output,
      'input_2:0': enc_output,
      'input_3:0': tf.tensor(false, [], 'bool'),
      'input_4:0': combined_mask,
      //'input_3:0': training_tensor, //tf.tensor([false]),
      //training_tensor,
      //tf.tensor(inputs[2]),
      'input_5:0': dec_padding_mask,
      //tf.tensor(tmp, [40, 1, 1, 40]),
    });
    predictions = predictions.slice([0, predictions.shape[1] - 1, 0], [predictions.shape[0], -1, -1]);
    predicted_id = tf.cast(tf.argMax(predictions, -1), 'int32');
    if (i < prefix_ids.length) {
      predicted_id = tf.tensor(prefix_ids[i], [1,1], 'int32');
    }
    var predicted_id_num = predicted_id.dataSync()[0]
    if (predicted_id_num == tok_en.vocab_size+1) {
      return tf.squeeze(output, 0);
    }
    output = tf.concat([output, predicted_id], -1);
  }
  return tf.squeeze(output, 0);
}

var cached_encoder_outputs = {}

async function translate(text, prefix) {
  var enc_output = cached_encoder_outputs[text]
  if (enc_output === undefined) {
    enc_output = await precompute_encoder_output(text);
    cached_encoder_outputs[text] = enc_output
  }
  //console.log('enc_output is:');
  //enc_output.print()
  var decoded_tokens = await evaluate(text, prefix, enc_output);
  //console.log('decoded tokens returnd')
  //console.log(decoded_tokens)
  var tokens_list = [];
  decoded_tokens.dataSync().forEach((x) => {
    if (x < tok_en.vocab_size) {
      tokens_list.push(x);
    }
  })
  var out_text = tok_en.ids_to_text(tokens_list);
  //console.log(tokens_list);
  //console.log(out_text);
  return out_text
}

// async function update_shown_translation() {
//   var newsrc = document.querySelector('#srctxt').value
//   var prefix = document.querySelector('#prefix').value
//   var translation = await translate(newsrc, prefix);
//   document.querySelector('#suggestion').innerText = translation
// }

async function update_translations() {
  var prev_newsrc = null;
  var prev_prefix = null;
  while (true) {
    await new Promise(function(resolve, reject) {
      setTimeout(resolve, 100);
    })
    var newsrc = document.querySelector('#srctxt').value
    var prefix = document.querySelector('#prefix').value
    if (newsrc === prev_newsrc && prefix === prev_prefix) {
      continue
    }
    if (!(prefix.endsWith(' ') || prefix === '')) {
      continue
    }
    prev_newsrc = newsrc
    prev_prefix = prefix
    var translation = await translate(newsrc, prefix);
    document.querySelector('#suggestion').innerText = translation;
  }
}

// setInterval(async function() {
//   update_shown_translation();
// }, 1000);

async function main() {
  await loadTransformerLayersModel();
  //document.querySelector('#srctxt').onkeyup = update_shown_translation;
  //document.querySelector('#prefix').onkeyup = update_shown_translation;
  document.querySelector('#srctxt').value = 'este é um problema que temos que resolver.'
  //update_shown_translation()
  update_translations()
}

main();