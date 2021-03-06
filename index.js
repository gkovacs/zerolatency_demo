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
  document.querySelector('#srctxt').value = 'este ?? um problema que temos que resolver.'
  //update_shown_translation()
  update_translations()
}

main();