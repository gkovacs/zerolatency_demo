var vocabulary = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789';

function encode_idx_to_chars(outputs) {
  var output_chars = [];
  for (var output of outputs) {
    var chars_list = output.map(x => vocabulary[x]);
    output_chars.push(chars_list.join(''));
  }
  return output_chars;
}

function convert_to_char_to_word(idx_to_word) {
  var output = {};
  for (var i = 0; i < idx_to_word.length; ++i) {
    var word = idx_to_word[i];
    var char = vocabulary[i];
    output[char] = word;
  }
  return output;
}

function encode_as_chars(arr_list) {
  var word_to_idx = {}
  var idx_to_word = []
  var outputs = [];
  for (var arr of arr_list) {
    var cur_output = [];
    for (var word of arr) {
      if (word_to_idx[word] === undefined) {
        var cur_idx = idx_to_word.length;
        word_to_idx[word] = cur_idx;
        idx_to_word.push(word);
        cur_output.push(cur_idx);
      } else {
        var cur_idx = word_to_idx[word];
        cur_output.push(cur_idx);
      }
    }
    outputs.push(cur_output);
  }
  return {arrs: encode_idx_to_chars(outputs), char_to_word: convert_to_char_to_word(idx_to_word)};
}

function compute_diff_for_arrays(arr_a, arr_b) {
  var {arrs, char_to_word} = encode_as_chars([arr_a, arr_b])
  var diff_output = diff(arrs[0], arrs[1]);
  var new_output = [];
  for (var [type, chars] of diff_output) {
    words = chars.split('').map(x => char_to_word[x]);
    new_output.push([type, words]);
  }
  return new_output
}
