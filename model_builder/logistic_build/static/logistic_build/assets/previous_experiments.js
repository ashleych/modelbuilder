// https://codepen.io/ashleych82/pen/WNKrdpp Source
$(document).ready(function () {
console.log("objecinitiat");
$("#advance").on("click", function() {
  var $bar = $(".ProgressBar");
  if ($bar.children(".is-current").length > 0) {
    $bar.children(".is-current").removeClass("is-current").addClass("is-complete").next().addClass("is-current");
  } else {
    $bar.children().first().addClass("is-current");
  }
});

$("#previous").on("click", function() {
  var $bar = $(".ProgressBar");
  if ($bar.children(".is-current").length > 0) {
    $bar.children(".is-current").removeClass("is-current").prev().removeClass("is-complete").addClass("is-current");
  } else {
    $bar.children(".is-complete").last().removeClass("is-complete").addClass("is-current");
  }
});

});


// $(document).ready(function() {
//   $("#search-button").click(function() {
//       console.log('hello');
//   });
// });