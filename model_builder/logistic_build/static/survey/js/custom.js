$(document).ready(function () {
  // Look for an input with date css class and load flatpickr on it, if it exists
  const dateInputFields = $("input.date");
  if (dateInputFields.length > 0) {
    dateInputFields.flatpickr();
  }
});

// document.addEventListener("DOMContentLoaded", function() {
//   $('#companyUpdOK').toast();
// });

// document.addEventListener("DOMContentReady", function() {

//   $("#mytoast").toast();
//   $('.toast').toast('show');

// });

$(document).ready(function () {
  $("body").on("click", ".close", function () {
    // $(this).closest('.toast').toast('hide')
    $(".toast").hide();
  });
});


