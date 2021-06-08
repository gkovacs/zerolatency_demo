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
    var text_full = document.querySelector('#prefix').value
    var prefix_end_idx = document.querySelector('#prefix').selectionStart;
    var text_typed_after_cursor_words = text_full.substr(prefix_end_idx).split(' ').filter(x => x.length > 0);
    window.text_typed_after_cursor_words = text_typed_after_cursor_words;
    var prefix = text_full.substr(0, prefix_end_idx);
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
    word_block.style.color = 'white'
    document.querySelector('#suggestion').append(word_block)
    var translation_after = translation.substr(prefix.length)
    var translation_after_words = translation_after.split(' ')
    window.translation_after_words = translation_after_words
    //var prefix_within_output = 0;
    var diff_list = compute_diff_for_arrays(text_typed_after_cursor_words, translation_after_words)
    var i = 0;
    for (var [diff_type, word_list] of diff_list) {
      if (diff_type == -1) {
        // deletion in diff
        continue;
      }
      for (var word of word_list) {
        var word_block = document.createElement('span');
        word_block.setAttribute('display', 'inline-block')
        word_block.setAttribute('idx', i);
        word_block.className = 'suggestion'
        word_block.setAttribute('id', 'suggestion_' + i);
        word_block.setAttribute('stext', word);
        word_block.innerText = word + ' ';
        //if (i === 0) {
        //  word_block.style.backgroundColor = 'lightblue';
        //}
        if (diff_type === 1) {
          word_block.style.backgroundColor = 'lightgreen';
        } else {
          word_block.style.color = 'grey';
        }
        (function(i) {
          word_block.onmouseover = function() {
            console.log('word ' + i + ' is being responded to');
            //setHoveredIdx(i);
          }
          word_block.onmouseleave = function() {
            //setHoveredIdx(0);
          }
          word_block.onmousedown = function(evt) {
            console.log('word ' + i + ' was clicked on');
            completeWordsUntil(i);
            document.querySelector('#prefix').focus()
            evt.preventDefault();
          }
        })(i);
        document.querySelector('#suggestion').append(word_block)
        i += 1;
      }
    }
    // for (var i = 0; i < translation_after_words.length; ++i) {
    //   var word = translation_after_words[i];
    //   //var is_before_cursor = prefix_within_output < prefix_end_idx;
    //   //prefix_within_output += word.length;
    //   var word_block = document.createElement('span');
    //   word_block.setAttribute('display', 'inline-block')
    //   word_block.setAttribute('idx', i);
    //   word_block.className = 'suggestion'
    //   word_block.setAttribute('id', 'suggestion_' + i);
    //   word_block.setAttribute('stext', word);
    //   word_block.innerText = word + ' ';
    //   if (i === 0) {
    //     word_block.style.backgroundColor = 'lightblue';
    //   }
    //   (function(i) {
    //     word_block.onmouseover = function() {
    //       console.log('word ' + i + ' is being responded to');
    //       setHoveredIdx(i);
    //     }
    //     word_block.onmouseleave = function() {
    //       setHoveredIdx(0);
    //     }
    //     word_block.onmousedown = function(evt) {
    //       console.log('word ' + i + ' was clicked on');
    //       completeWordsUntil(i);
    //       document.querySelector('#prefix').focus()
    //       evt.preventDefault();
    //     }
    //   })(i);
    //   document.querySelector('#suggestion').append(word_block)
      
    // }
  }
}

function completeWordsUntil(idx) {
  var words_to_complete = window.translation_after_words.slice(0, idx + 1);
  var text_to_complete = words_to_complete.join(' ');
  var prefix_end_idx = document.querySelector('#prefix').selectionStart;
  var old_text = document.querySelector('#prefix').value;
  var part_before = old_text.substr(0, prefix_end_idx);
  var part_after = old_text.substr(prefix_end_idx);
  document.querySelector('#prefix').value = part_before + text_to_complete + ' ' + part_after;
  document.querySelector('#prefix').selectionStart = prefix_end_idx + text_to_complete.length + 1;
  document.querySelector('#prefix').selectionEnd = prefix_end_idx + text_to_complete.length + 1;

}

// setInterval(async function() {
//   update_shown_translation();
// }, 1000);

// function cursorPositionChanged(index_in_output) {
//   var newsrc = document.querySelector('#srctxt').value;
//   var full_text = document.querySelector('#prefix').value;
//   var prefix = full_text.substr(0, index_in_output);
//   console.log('prefix is')
//   console.log(prefix)
//   console.log('full text is')
//   console.log(full_text)
//   console.log('newsrc is')
//   console.log(newsrc)
// }

// function cursorPositionChangedHandler(evt) {
//   cursorPositionChanged(evt.target.selectionStart)
// }

async function main() {
  //await tf.setBackend('wasm');
  await tf.setBackend('webgl')
  await loadTransformerLayersModel();
  //document.querySelector('#srctxt').onkeyup = update_shown_translation;
  //document.querySelector('#prefix').onkeyup = update_shown_translation;
  //document.querySelector('#prefix').addEventListener('focus', positionChanged);
  //document.querySelector('#prefix').addEventListener('click', cursorPositionChangedHandler);
  //document.querySelector('#prefix').addEventListener('keydown', cursorPositionChangedHandler);
  document.querySelector('#prefix').addEventListener('keydown', function(evt) {
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
  });
  document.querySelector('#srctxt').value = 'este é um problema que temos que resolver.'
  //update_shown_translation()
  update_translations()
}

main();