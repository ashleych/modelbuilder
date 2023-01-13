function mark_as_read() {
  // url='/logistic_build/notificationmodelbuild/mark_as_read'
  // url = 
  // var form = $(this);
  url=$('#markasread').attr('data-url');
  // url=form.attr("data-validate-username-url");
  $.ajax({
    url: url,
    type: "GET",
    contentType: 'application/json; charset=utf-8',
    // dataType: "json",
    dataType: 'text json',
    success: (data) => {
      console.log("mark as rad called ")
      window.location.reload();
      // const todoList = $("#todoList");
      // todoList.empty();
      
      // (data.context).forEach(todo => {
        //   const todoHTMLElement = `
        //     <li>
        //       <p>Task: ${todo.task}</p>
        //       <p>Completed?: ${todo.completed}</p>
        //     </li>`
        //   todoList.append(todoHTMLElement);
        // });
      },
      error: (error) => {
        window.location.reload();
        console.log(error);
      }
    });
  }
  function notifications_delete() {
    console.log('ajax called');
    // url='http://127.0.0.1:8000/logistic_build/notificationmodelbuild/delete'
    url = '{%url "notificationmodelbuild_delete" %}' ;
    url=$('#delete').attr('data-url');
    $.ajax({
      url: url,
      type: "GET",
      // dataType: "json",
      dataType: 'text json',
      success: (data) => {
        console.log("success clearingt");
        window.location.reload();
        location.reload();
      },
      error: (error) => {
        window.location.reload();
        console.log(error);
      }
    });
  }