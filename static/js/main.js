$(document).ready(function() {
    $("#solve").on('click', function() {
        $.ajax({
            url: '/solve',
            dataType: 'json',
            type: 'post',
            contentType: 'application/json',
            data: JSON.stringify( { 'algorithm': $('#algorithm').val(), 'data': $('#data').val() }),
            processData: false,
            success: function(data, textStatus, jQxhr) {
                console.log("resuest done");
            },
            error: function(jQxhr, textStatus, errorThrow) {
                // MAKE ERROR Appear
            }
        });    
    });
}); 








