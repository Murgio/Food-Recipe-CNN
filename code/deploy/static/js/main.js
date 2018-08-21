$(document).ready(function () {
    // Init
    $('.image-section').hide();
    $('.loader').hide();
    $('#result').hide();
    //$('#header').hide();
    //$('.wrapper').hide();

    // Upload Preview
    function readURL(input) {
        if (input.files && input.files[0]) {
            var reader = new FileReader();
            reader.onload = function (e) {
                $('#imagePreview').css('background-image', 'url(' + e.target.result + ')');
                $('#imagePreview').hide();
                $('#imagePreview').fadeIn(250);
            }
            reader.readAsDataURL(input.files[0]);
        }
    }
    $("#imageUpload").change(function () {
        $('.image-section').show();
        $('#btn-predict').show();
        $('#result').text('');
        $('#result').hide();
        readURL(this);
    });

    // Predict
    $('#btn-predict').click(function () {
        var form_data = new FormData($('#upload-file')[0]);
        // Show loading animation
        $(this).hide();
        $('.loader').show();
        $('#result').fadeIn(50);

        // Make prediction by calling api /predict
        $.ajax({
            type: 'POST',
            url: '/predict',
            data: form_data,
            contentType: false,
            cache: false,
            processData: false,
            async: true,
            success: function (data) {
                // Get and display the result
                $('.loader').hide();
                $('#result').append(data);
                console.log('Success!');
                close_instructions()
                let images = document.querySelectorAll("img");
                lazyload(images);
            },
        });
    });
});

function close_instructions()
{
    var x = document.querySelectorAll('.cont_modal');
    var i;
    for (i = 0; i < x.length; i++) {
        x[i].className = "cont_modal";
    }
}
var c = 0;
function open_close(){

    var x = document.querySelectorAll('.cont_modal');

    if(c % 2 == 0){
        var i;
        for (i = 0; i < x.length; i++) {
            x[i].className = "cont_modal cont_modal_active";
        }
        c++;
    }else {
        var i;
        for (i = 0; i < x.length; i++) {
            x[i].className = "cont_modal";
        }
        c++;
    }
}
