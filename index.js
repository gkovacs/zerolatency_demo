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
        } else if (tokenizer_en_def.model.vocab[cur_text.replace(' ', '▁')] != undefined) {
          output.push(tokenizer_en_def.model.vocab[cur_text.replace(' ', '▁')])
          cur_pos += token_length
          break
        } else if (tokenizer_en_def.model.vocab['▁' + cur_text] != undefined) {
          output.push(tokenizer_en_def.model.vocab['▁' + cur_text])
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
  
  function ids_to_text_list(token_id_list) {
    var output = []
    for (var i = 0; i < token_id_list.length; ++i) {
      var token_id = token_id_list[i];
      var token = id_to_vocab[token_id];
      if (token[0] === '▁') {
        if (i === 0) {
          token = token.substr(1)
        } else {
          token = ' ' + token.substr(1)
        }
      }
      output.push(token)
    }
    return output
  }
  
  function ids_to_text(token_id_list) {
    var output = []
    for (var i = 0; i < token_id_list.length; ++i) {
      var token_id = token_id_list[i];
      var token = id_to_vocab[token_id];
      if (token[0] === '▁') {
        if (i === 0) {
          token = token.substr(1)
        } else {
          token = ' ' + token.substr(1)
        }
      }
      output.push(token)
    }
    return output.join('');
  }

  var vocab_size = Object.keys(tokenizer_en_def.model.vocab).length;

  return {
    'text_to_ids': text_to_ids,
    'ids_to_text': ids_to_text,
    'ids_to_text_list': ids_to_text_list,
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
  //var top_predictions_list = tf.expandDims(tf.expandDims(decoder_input, 0), 0);
  //console.log('top_predictions_list is')
  //top_predictions_list.print()
  console.log('output dimensions are')
  output.print()
  var dec_padding_mask = create_padding_mask(encoder_input);
  var prefix_ids = [];
  var prefix_tokens = tok_en.text_to_ids(prefix);
  for (var token_id of prefix_tokens) {
    prefix_ids.push(token_id);
  }
  var top_predictions = null;
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
    //console.log('predicted_id is')
    //predicted_id.print()
    //console.log('top 2 predicted ids are:')
    //top_predictions.print()
    //top_predictions_list = tf.concat([top_predictions_list, top_predictions]);
    //console.log('top predictions list is:')
    //top_predictions_list.print();
    if (i < prefix_ids.length) {
      predicted_id = tf.tensor(prefix_ids[i], [1,1], 'int32');
    } else if (i === prefix_ids.length) {
      top_predictions = predictions.topk(1000, true).indices;
    }
    var predicted_id_num = predicted_id.dataSync()[0]
    if (predicted_id_num == tok_en.vocab_size+1) {
      console.log('output pre-squeeze 1 is:')
      output.print()
      return [tf.squeeze(output, 0), top_predictions];
    }
    output = tf.concat([output, predicted_id], -1);
  }
  console.log('output pre-squeeze 2 is:')
  output.print()
  return [tf.squeeze(output, 0), top_predictions];
}

var cached_encoder_outputs = {}

var cached_translate_results = {}

async function translate(text, prefix) {
  if (text[0] != ' ') {
    text = ' ' + text
  }
  prefix = prefix.trim();
  if (cached_translate_results[text] !== undefined) {
    if (cached_translate_results[text][prefix] !== undefined) {
      return cached_translate_results[text][prefix]
    }
  } else {
    cached_translate_results[text] = {}
  }
  var enc_output = cached_encoder_outputs[text]
  if (enc_output === undefined) {
    enc_output = await precompute_encoder_output(text);
    cached_encoder_outputs[text] = enc_output
  }
  //console.log('enc_output is:');
  //enc_output.print()
  var [decoded_tokens, top_predictions] = await evaluate(text, prefix, enc_output);
  //console.log('decoded tokens returnd')
  //console.log(decoded_tokens)
  var top_predictions_ids = [];
  top_predictions.dataSync().forEach((x) => {
    if (x < tok_en.vocab_size) {
      top_predictions_ids.push(x);
    }
  })
  var top_predictions_text = tok_en.ids_to_text_list(top_predictions_ids);
  var tokens_list = [];
  decoded_tokens.dataSync().forEach((x) => {
    if (x < tok_en.vocab_size) {
      tokens_list.push(x);
    }
  })
  var out_text = tok_en.ids_to_text(tokens_list);
  //console.log(tokens_list);
  //console.log(out_text);
  var output = [out_text, top_predictions_text]
  cached_translate_results[text][prefix] = output;
  return output;
}

// async function update_shown_translation() {
//   var newsrc = document.querySelector('#srctxt').value
//   var prefix = document.querySelector('#prefix').value
//   var translation = await translate(newsrc, prefix);
//   document.querySelector('#suggestion').innerText = translation
// }

function setHoveredIdx(idx) {
  for (var i = 0; i < window.translation_after_words.length; ++i) {
    var is_highlighted = i <= idx;
    var word_block = document.querySelector('#suggestion_' + i);
    if (is_highlighted) {
      word_block.style.backgroundColor = 'lightblue'
    } else {
      word_block.style.backgroundColor = 'white'
    }
  }
}

function set_alternatives_list(prefix_so_far) {
  document.querySelector('#alternatives').innerText = '';
  var num_candidates_shown = 0;
  for (var alternative_text of window.alternatives_list) {
    alternative_text = alternative_text.trim();
    if (!alternative_text.startsWith(prefix_so_far)) {
      continue;
    }
    var word_block = document.createElement('span')
    word_block.innerText = alternative_text + ' ';
    word_block.setAttribute('stext', alternatives_list);
    if (num_candidates_shown === 0) {
      word_block.style.backgroundColor = 'lightblue';
    }
    document.querySelector('#alternatives').append(word_block);
    num_candidates_shown += 1;
    if (num_candidates_shown >= 5) {
      break;
    }
  }
}

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
    prev_newsrc = newsrc
    prev_prefix = prefix
    //if (prefix !== '' && !(prefix.endsWith(' '))) {
    //  var word_typed_so_far = prefix.substr(prefix.lastIndexOf(' ') + 1);
    //  set_alternatives_list(word_typed_so_far);
    //  continue
    //}
    var partial_word = '';
    var prefix_to_translate = '';
    if (prefix.endsWith(' ')) {
      partial_word = ''
    } else {
      partial_word = prefix.substr(prefix.lastIndexOf(' ') + 1);
      prefix = prefix.substr(0, prefix.lastIndexOf(' ') + 1);
    }
    var [translation, alternatives_list] = await translate(newsrc, prefix);
    window.alternatives_list = alternatives_list;
    set_alternatives_list(partial_word);
    document.querySelector('#suggestion').innerText = '';
    var translation_before = prefix;
    var word_block = document.createElement('span')
    word_block.innerText = translation_before
    word_block.style.color = 'grey'
    document.querySelector('#suggestion').append(word_block)
    var translation_after = translation.substr(prefix.length)
    var translation_after_words = translation_after.split(' ')
    window.translation_after_words = translation_after_words
    for (var i = 0; i < translation_after_words.length; ++i) {
      var word = translation_after_words[i];
      var word_block = document.createElement('span');
      word_block.setAttribute('display', 'inline-block')
      word_block.setAttribute('idx', i);
      word_block.className = 'suggestion'
      word_block.setAttribute('id', 'suggestion_' + i);
      word_block.setAttribute('stext', word);
      word_block.innerText = word + ' ';
      if (i === 0) {
        word_block.style.backgroundColor = 'lightblue';
      }
      (function(i) {
        word_block.onmouseover = function() {
          console.log('word ' + i + ' is being responded to');
          setHoveredIdx(i);
        }
        word_block.onmouseleave = function() {
          setHoveredIdx(0);
        }
        word_block.onmousedown = function(evt) {
          console.log('word ' + i + ' was clicked on');
          completeWordsUntil(i);
          document.querySelector('#prefix').focus()
          evt.preventDefault();
        }
      })(i);
      document.querySelector('#suggestion').append(word_block)
      
    }
  }
}

function completeWordsUntil(idx) {
  var words_to_complete = window.translation_after_words.slice(0, idx + 1);
  var text_to_complete = words_to_complete.join(' ');
  document.querySelector('#prefix').value += text_to_complete + ' ';

}

// setInterval(async function() {
//   update_shown_translation();
// }, 1000);

function toggleCompletionHotkeys() {
  if (window.completion_hotkeys_shown ) {
    hideCompletionHotkeys();
    
  } else {
    showCompletionHotkeys();
    window.completion_hotkeys_shown = true;
  }
}

function hideCompletionHotkeys() {
  for (var i = 0; i < window.translation_after_words.length; ++i) {
    //var is_highlighted = i <= idx;
    var word_block = document.querySelector('#suggestion_' + i);
    if (word_block._tippy) {
      word_block._tippy.unmount();
    }
    //word_block.innerHTML = word_block.getAttribute('stext') + ' ';
    // var child_block = document.createElement('div');
    // child_block.innerText = i;
    // child_block.style.position = 'relative';
    // word_block.append(child_block);
  }
  window.completion_hotkeys_shown = false;
}

function showCompletionHotkeys() {
  window.hotkey_list = []
  window.hotkey_to_idx = {}
  for (var i = 0; i < window.translation_after_words.length; ++i) {
    //var is_highlighted = i <= idx;
    var hotkey = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'[i];
    var word_block = document.querySelector('#suggestion_' + i);
    tippy('#suggestion_' + i, {content: hotkey, arrow: true})[0].show()
    //word_block.setAttribute('hotkey_' + hotkey, true);
    hotkey_list.push(hotkey)
    hotkey_to_idx[hotkey] = i;
    //word_block.innerHTML = '<span style="background-color: yellow">' + hotkey + '</span>' + word_block.getAttribute('stext') + ' ';
    // var child_block = document.createElement('div');
    // child_block.innerText = i;
    // child_block.style.position = 'relative';
    // word_block.append(child_block);
  }
  window.completion_hotkeys_shown = true;
}

async function main() {
  //await tf.setBackend('wasm');
  await tf.setBackend('webgl')
  await loadTransformerLayersModel();
  //document.querySelector('#srctxt').onkeyup = update_shown_translation;
  //document.querySelector('#prefix').onkeyup = update_shown_translation;
  document.querySelector('#prefix').onkeydown = function(evt) {
    if (evt.which === 13) {
      // enter
      completeWordsUntil(0);
      //var next_word = window.translation_after_words[0]
      //document.querySelector('#prefix').value = document.querySelector('#prefix').value + next_word + ' '
      evt.preventDefault();
      return;
    }
    if (evt.which === 9) {
      // tab
      toggleCompletionHotkeys();
      //showCompletionHotkeys();
      evt.preventDefault();
      return;
    }
    if (window.completion_hotkeys_shown) {
      var hotkey = evt.key.toUpperCase();
      var idx = window.hotkey_to_idx[hotkey]
      hideCompletionHotkeys();
      if (idx !== undefined) {
        console.log('evt.key is: ' + evt.key + ' and idx is ' + idx)
        //var word_block = document.querySelector('#suggestion_' + idx);
        //var idx = word_block.getAttribute('idx');
        completeWordsUntil(idx);
        evt.preventDefault();
        return;
      }
    }
  }
  document.querySelector('#srctxt').value = 'este é um problema que temos que resolver.'
  //update_shown_translation()
  update_translations()
}

main();